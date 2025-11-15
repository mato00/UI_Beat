from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
import torch
import wfdb

import dataset.ecg_preprocess as ep
import utils.qrs_post_process as pp
from models.multi_head import decoder4qrs, encoder4qrs, phi_qrs
from models.qrs_model import QRSModel

np.set_printoptions(threshold=np.inf)

BATCH = 100


def load_pytorch_model(model_path: Path, device: torch.device) -> QRSModel:
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)

    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _prepare_model_input(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 1:
        return data.reshape(1, 1, -1)
    if data.ndim == 2:
        if data.shape[0] == 1 or data.shape[1] == 1:
            flattened = data.reshape(-1)
            return flattened.reshape(1, 1, -1)
        return data[:, np.newaxis, :]
    if data.ndim == 3:
        if data.shape[-1] == 1:
            data = data[..., 0]
        if data.shape[1] == 1:
            return data
        return data[:, np.newaxis, :]
    raise ValueError(f"Unsupported ECG input shape: {data.shape}")


def run_model(model: QRSModel, arr: np.ndarray, device: torch.device) -> np.ndarray:
    batch = _prepare_model_input(arr)
    tensor = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        logits = model(tensor, return_projection=True)
    return logits.cpu().numpy()


def load_ecg(file_path: str, database: str) -> tuple[np.ndarray, int, int, int]:
    if database == 'cpsc2019':
        sig = sio.loadmat(file_path)['ecg'].squeeze()
        fs = 500
        length = len(sig)
        lead_num = 1
    elif database in {'mitdb', 'incart', 'ludb', 'qt'}:
        sig, fields = wfdb.rdsamp(file_path)
        length = len(sig)
        fs = fields['fs']
        lead_num = np.shape(sig)[-1]
    else:
        raise ValueError('Unsupported database for unified testing platform.')

    return sig, length, fs, lead_num


def ngrams(data: np.ndarray, length: int, fs: int) -> list[np.ndarray]:
    grams = []
    for i in range(0, length - fs * 10, fs * 6):
        grams.append(data[i: i + fs * 10])
    return grams


def qrs_detect(
    database: str,
    data_path: str,
    ans_path: str,
    model_path: Path,
    is_multi_leads: bool = False,
    device: Optional[torch.device] = None,
) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pytorch_model(Path(model_path), device)

    records_file = Path(data_path) / 'RECORDS'
    files = records_file.read_text().splitlines()

    os.makedirs(ans_path, exist_ok=True)

    for file in files:
        print(file)
        if database == 'cpsc2019':
            file_path = os.path.join(data_path, file + '.mat')
            ecg, length, fs, _ = load_ecg(file_path, database)

            ecg_test = ep.ecg_process(ecg, fs)

            logits = run_model(model, ecg_test, device)
            logits = np.around(logits, decimals=2)
            logits = np.squeeze(logits)

            r_ans = pp.correct(logits[:, 0])
            r_ans = r_ans.astype(float)
            r_ans *= (fs / 62.5)
            r_ans = np.trunc(r_ans)

            np.save(os.path.join(ans_path, f'{file}.npy'), r_ans)

        elif database == 'ludb':
            file_path = os.path.join(data_path, file)
            ecg, length, fs, lead_num = load_ecg(file_path, database)

            uncertain_leads = []
            logits_leads = []
            for j in range(lead_num):
                ecg_singleL = ecg[:, j]
                ecg_test = ep.ecg_process(ecg_singleL, fs)

                logits = run_model(model, ecg_test, device)
                logits = np.around(logits, decimals=2)
                logits = np.squeeze(logits)

                if not is_multi_leads:
                    r_ans = pp.correct(logits[:, 0])
                    r_ans = r_ans.astype(float)
                    r_ans *= (fs / 62.5)
                    r_ans = np.trunc(r_ans)

                    np.save(os.path.join(ans_path, f'{file}_{j}.npy'), r_ans)
                else:
                    uncertain_score = pp.uncertain_est(logits, alpha=.5)
                    uncertain_leads.append(uncertain_score)
                    logits_leads.append(logits[:, 0])

            if is_multi_leads:
                uncertain_leads = np.array(uncertain_leads).T
                logits_leads = np.array(logits_leads).T

                preds_ua = pp.multi_lead_select(logits_leads, uncertain_leads)
                r_ans = pp.correct(preds_ua)
                r_ans = r_ans.astype(float)
                r_ans *= (fs / 62.5)
                r_ans = np.trunc(r_ans)

                np.save(os.path.join(ans_path, f'{file}_0.npy'), r_ans)

        else:
            file_path = os.path.join(data_path, file)
            ecg, length, fs, lead_num = load_ecg(file_path, database)

            uncertain_leads = []
            logits_leads = []
            for j in range(lead_num):
                ecg_singleL = ecg[:, j]
                ecg_batch = ngrams(ecg_singleL, length, fs)
                ecg_batch = np.array(ecg_batch)
                ecg_test = ep.ecg_process_batch(ecg_batch, fs)

                preds = []
                segment_count = len(ecg_test)
                full_batches = segment_count // BATCH
                pred_remain = None

                for i in range(full_batches):
                    ecg_b = ecg_test[i * BATCH: (i + 1) * BATCH]
                    logits = run_model(model, ecg_b, device)
                    logits = np.around(logits, decimals=2)
                    preds.extend(logits)

                remainder = segment_count - BATCH * full_batches
                if remainder > 0:
                    ecg_b = ecg_test[BATCH * full_batches:]
                    logits = run_model(model, ecg_b, device)
                    logits = np.around(logits, decimals=2)
                    preds.extend(logits)

                    ecg_last = ecg_singleL[-fs * 10:]
                    ecg_last = ep.ecg_process(ecg_last, fs)

                    logits_last = run_model(model, ecg_last, device)
                    logits_last = np.around(logits_last, decimals=2)

                    gram_len = fs * 6 * (len(ecg_batch) - 1) + fs * 8
                    length_remain = (length - gram_len) / fs
                    pred_remain = logits_last[0, -1 * int(length_remain * 62.5):]

                preds = np.array(preds)
                if preds.size == 0:
                    continue

                pred_init = preds[0, :500, :]
                if preds.shape[0] > 1:
                    pred_post = preds[1:, 125:500, :]
                    pred_post = np.reshape(pred_post, (pred_post.shape[0] * pred_post.shape[1], -1))
                    combined = np.concatenate([pred_init, pred_post], axis=0)
                else:
                    combined = pred_init

                if pred_remain is not None and pred_remain.size > 0:
                    combined = np.concatenate([combined, pred_remain], axis=0)

                if not is_multi_leads:
                    r_ans = pp.correct(combined[:, 0])
                    r_ans = r_ans.astype(float)
                    r_ans *= (fs / 62.5)
                    r_ans = np.trunc(r_ans)

                    np.save(os.path.join(ans_path, f'{file}_{j}.npy'), r_ans)
                else:
                    uncertain_score = pp.uncertain_est(combined, alpha=.5)
                    uncertain_leads.append(uncertain_score)
                    logits_leads.append(combined[:, 0])

            if is_multi_leads:
                uncertain_leads = np.array(uncertain_leads).T
                logits_leads = np.array(logits_leads).T

                preds_ua = pp.multi_lead_select(logits_leads, uncertain_leads)
                r_ans = pp.correct(preds_ua)
                r_ans = r_ans.astype(float)
                r_ans *= (fs / 62.5)
                r_ans = np.trunc(r_ans)

                np.save(os.path.join(ans_path, f'{file}_0.npy'), r_ans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--database_type', required=True, help='database type')
    parser.add_argument('-d', '--data_path', required=True, help='path saving test record file')
    parser.add_argument('-r', '--result_save_path', required=True, help='path saving QRS-complex location results')
    parser.add_argument('--model-path', default=str(Path(__file__).resolve().parents[1] / 'experiments' / 'qrs_detect' / 'saved_model' / 'best_model.pt'), help='path to the saved PyTorch best model')
    parser.add_argument('--device', default=None, help='device to run inference on (e.g., cpu, cuda:0)')
    parser.add_argument('-ml', '--is_multi_leads', action='store_true', help='flag for multi-lead processing')

    args = parser.parse_args()

    device_arg = torch.device(args.device) if args.device else None
    qrs_detect(
        database=args.database_type,
        data_path=args.data_path,
        ans_path=args.result_save_path,
        model_path=Path(args.model_path),
        is_multi_leads=bool(args.is_multi_leads),
        device=device_arg,
    )
