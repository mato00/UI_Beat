import numpy as np
import math


def correct(result, uc, is_bce=False):
    if is_bce:
        pos = np.argwhere(result >= .5).flatten() 
    else:
        pos = np.argwhere(result >= 0).flatten()
    diff_pos = np.diff(pos)

    diff_uc = np.diff(uc)

    rpos = []
    pre = 0
    last = len(pos)-1
    cand_poses = np.where(diff_pos >= 4)[0]
    if last > 1:
        for index, cand_pos in enumerate(cand_poses):
            if cand_pos - pre >= 4:
                r_cand = round((pos[pre] + pos[cand_pos]) / 2)
                # print (r_cand)
                rpos.append(r_cand)
            pre = cand_pos + 1

        rpos.append(round((pos[pre] + pos[last]) / 2))
        qrs = np.array(rpos)
        qrs = qrs.astype(int)
    else:
        qrs = np.array([])

    qrs_diff = np.diff(qrs)
    check = True
    r = 0
    while check:
        qrs_diff = np.diff(qrs)
        if len(qrs_diff[qrs_diff <= 12]) == 0:
            check = False
            break
        while r < len(qrs_diff):
            if qrs_diff[r] <= 12:
                prev = qrs[r]
                next = qrs[r+1]
                qrs = np.delete(qrs,r)
                qrs = np.delete(qrs,r)
                qrs = np.insert(qrs, r, (prev+next)/2)
                qrs_diff = np.diff(qrs)
            r += 1
        check = False
    
    if np.mean(uc) >= 10:
        noise_starts = np.array([0])
        noise_ends = np.array([len(uc)-1])
    else:
        noise_starts = np.argwhere(diff_uc >= 5).flatten() + 1
        noise_ends = np.argwhere(diff_uc <= -5).flatten() + 1

        if uc[0] >= 5:
            noise_starts = np.insert(noise_starts, 0, 0)
        if uc[-1] >= 5:
            noise_ends = np.insert(noise_ends, len(noise_ends), len(uc)-1)
    
    noise_starts = np.expand_dims(noise_starts, -1)
    noise_ends = np.expand_dims(noise_ends, -1)
    
    noise_idx = np.concatenate((noise_starts, noise_ends), axis=-1)

    return qrs, noise_idx

def entropy(p):
    logp = np.log(p)
    plogp = p * logp
    entropy = -np.sum(plogp, axis=-1)
    
    return entropy

def en_est(logits):
    logits_p = np.expand_dims(logits[:, :, 1], -1)
    logits_n = np.expand_dims(logits[:, :, 2], -1)
    logits_p_s = logits_p / (logits_p - logits_n + 1.)
    logits_n_s = (1. - logits_n) / (logits_p - logits_n + 1.)
    
    p = np.concatenate((logits_p_s, logits_n_s), -1)
    
    logp = np.log(p + 1e-5)
    plogp = p * logp
    entropy = -np.sum(plogp, axis=-1)
    
    return entropy


def mi_est(logits):
    logits_o = logits[:, :, 0]
    logits_p = logits[:, :, 1]
    logits_n = logits[:, :, 2]
    
    eu_p = logits_p * np.log(logits_p/(logits_o + 1e-5) + 1e-5)
    eu_n = (1. - logits_n) * np.log((1. - logits_n)/(1. - logits_o + 1e-5) + 1e-5)

    eu = eu_p + eu_n
    eu = np.mean(eu, axis=-1)
  
    return eu

def uncertain_est(logits, thr=0.12):
    
    au = en_est(logits)
    eu = mi_est(logits)

    ### noise screening ###
    eu[eu > thr] = 10
    ### noise screening ###

    eu = np.expand_dims(eu, 1)
    eu = np.repeat(eu, np.shape(logits)[1], axis=1)

    uncertain = eu + au
  
    return uncertain

def mc_dropout_uncertain_est(logits):
    uc = entropy(logits + 1e-6)

    return uc

def qrs_uncertain_score(logits, uncertain_seq):
    logits_o = logits[0, :, 0]
    qrs_o = correct(logits_o)
    
    def ngrams(arr, qrs_loc, length, step):
        grams = []
        for i in range(0, length-2, 1):
            start = qrs_loc[i]
            end = qrs_loc[i+2]
            grams.append(arr[start: end])

        return grams
    
    loc_screen = np.insert(qrs_o, len(qrs_o), len(logits_o))
    loc_screen = np.insert(loc_screen, 0, 0)
    total_beats_num = len(loc_screen)

    screen_periods = ngrams(uncertain_seq, loc_screen, total_beats_num, 1)
    uncertain_r = []
    for index, screen_period in enumerate(screen_periods):
        m = len(screen_period)
        n = np.sum(screen_period)
        r = n/math.log(m + 1e-6)
        uncertain_r.append(r)
    
    return uncertain_r, qrs_o

def multi_lead_select(logits_o_leads, uncertain_leads):
    lead_select = np.argmin(uncertain_leads, -1)
    
    uc = [uncertain_leads[i, lead_select[i]] for i in range(len(uncertain_leads))]
    logits_o = [logits_o_leads[i, lead_select[i]] for i in range(len(logits_o_leads))]
    logits_o = np.array(logits_o)

    return logits_o, uc