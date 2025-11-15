from __future__ import annotations

import argparse
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from pathlib import Path
from typing import Optional
import json
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch

import dataset.ecg_preprocess as ep
import utils.qrs_post_process as pp
from models.multi_head import decoder4qrs, encoder4qrs, phi_qrs
from models.qrs_model import QRSModel
from data.pwave_preprocess import extract_ccm


BATCH = 10


def load_pytorch_model(model_path: Path, device: torch.device) -> QRSModel:
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)

    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    return model

def ecg_process_batch(ecg, fs):
    ecgs = []
    for single_ecg in ecg:
        data_e = single_ecg.copy()
        data_e = data_e - ep.lowpass_filter(data_e, 0.5, fs)
        data_e = data_e - ep.highpass_filter(data_e, 75, fs)
        data_e -= np.mean(data_e)
        data_e = ep.downsample(data_e, fs, 200)
        
        data_e = preprocessing.scale(data_e)
        data_e = np.expand_dims(data_e, 0)

        ecgs.append(data_e)
    ecgs = np.array(ecgs)

    return ecgs

def ecg_process(ecg, fs):
    data_e = ecg.copy()
    data_e = data_e - ep.lowpass_filter(data_e, 0.5, fs)
    data_e = data_e - ep.highpass_filter(data_e, 75, fs)
    data_e -= np.mean(data_e)
    data_e = ep.downsample(data_e, fs, 200)
    
    data_e = preprocessing.scale(data_e)
    data_e = np.expand_dims(data_e, 0)
    data_e = np.expand_dims(data_e, 0)

    return data_e

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


def load_ecg(file_path: str, database: str):
    """
    Unified ECG loader.
    Adds support for CCM-based multi-lead P-wave data when database == 'p_wave'.
    Returns: sig(T, lead), length(T), fs, lead_num
    """

    if database == "ccm":
        records = extract_ccm(file_path)
        if not records:
            raise ValueError(f"No ECG segments extracted from CCM file: {file_path}")

        # First segment gives fs + lead_num
        fs0, seg0, pos0, maintype0, aftype0 = records[0]
        fs = int(fs0)
        lead_num = seg0.shape[1]

        segments = [seg0]

        # Concatenate remaining segments
        for fs_i, seg_i, pos_i, maintype_i, aftype_i in records[1:]:
            if int(fs_i) != fs:
                raise ValueError(f"Inconsistent sampling rate in CCM file {file_path}")
            if seg_i.shape[1] != lead_num:
                raise ValueError(f"Inconsistent lead count in CCM file {file_path}")
            segments.append(seg_i)

        # Stack into full ECG
        sig = np.vstack(segments)
        length = sig.shape[0]

        return sig, length, fs, lead_num


def ngrams(data: np.ndarray, length: int, fs: int) -> list[np.ndarray]:
    grams = []
    for i in range(0, length - fs * 10, fs * 6):
        grams.append(data[i: i + fs * 10])
    return grams


def pwave_detect(
    database: str,
    file_list: list[str],
    ans_path: str,
    model_path: Path,
    is_multi_leads: bool = True,
    device: Optional[torch.device] = None,
) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pytorch_model(Path(model_path), device)
    
    os.makedirs(ans_path, exist_ok=True)

    for file_path in file_list:
        print(file_path)
        file_name = os.path.basename(file_path)

        if database == 'ccm':
            ecg, length, fs, lead_num = load_ecg(file_path, database)
            print(f'fs: {fs}, lead num: {lead_num}, length: {length}, ecg shape: {ecg.shape}')

            uncertain_leads = []
            logits_leads = []
            for j in range(lead_num):
                ecg_singleL = ecg[:, j]
                ecg_batch = ngrams(ecg_singleL, length, fs)
                ecg_batch = np.array(ecg_batch)
                ecg_test = ecg_process_batch(ecg_batch, fs)

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
                    ecg_last = ecg_process(ecg_last, fs)

                    logits_last = run_model(model, ecg_last, device)
                    logits_last = np.around(logits_last, decimals=2)

                    gram_len = fs * 6 * (len(ecg_batch) - 1) + fs * 8
                    length_remain = (length - gram_len) / fs
                    pred_remain = logits_last[0, :, -1 * int(length_remain * 50):].squeeze()
                preds = np.array(preds).squeeze()
                if preds.size == 0:
                    continue

                pred_init = preds[0, :, :400]
                if preds.shape[0] > 1:
                    pred_post = preds[1:, :, 100: 400]
                    pred_post = np.reshape(pred_post, (pred_post.shape[1], pred_post.shape[0] * pred_post.shape[2]))
                    combined = np.concatenate([pred_init, pred_post], axis=1)
                else:
                    combined = pred_init
                if pred_remain is not None and pred_remain.size > 0:
                    combined = np.concatenate([combined, pred_remain], axis=1)
                    
                uncertain_score = pp.uncertain_est(combined, thr=0.12)
                uncertain_leads.append(uncertain_score)
                logits_leads.append(combined[0, :])

            print('Multi-lead P-wave detection processing...')
            uncertain_leads = np.array(uncertain_leads).T
            logits_leads = np.array(logits_leads).T
            preds_ua, uc = pp.multi_lead_select(logits_leads, uncertain_leads)
            r_ans = pp.correct(preds_ua, uc)
            r_ans = r_ans.astype(float)
            r_ans *= (fs / 50)
            r_ans = np.trunc(r_ans)
            print(f'Predicted P-wave locations: {r_ans}')
            np.save(os.path.join(ans_path, f'{file_name}_pwaves.npy'), r_ans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_type', default='ccm', help='database type')
    parser.add_argument('--result_save_path', default='experiments/nl_pwave_results/', help='path saving QRS-complex location results')
    parser.add_argument('--model-path', default='experiments/pwave_detection/logs/pwave_20251115_032436/models/best_model.pt', help='path to the saved PyTorch best model')
    parser.add_argument('--device', default=None, help='device to run inference on (e.g., cpu, cuda:0)')
    parser.add_argument('--is_multi_leads', default=True, help='flag for multi-lead processing')

    args = parser.parse_args()
    device_arg = torch.device(args.device) if args.device else None
    
    with open("data/p_wave_splits.json", "r") as f:
        splits = json.load(f)
    test_list = splits["test"]
    
    pwave_detect(
        database=args.database_type,
        file_list=test_list,
        ans_path=args.result_save_path,
        model_path=Path(args.model_path),
        is_multi_leads=args.is_multi_leads,
        device=device_arg,
    )
