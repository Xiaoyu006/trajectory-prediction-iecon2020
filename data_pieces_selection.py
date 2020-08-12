import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
import math

from lc_veh_selection import get_veh_id2traj, get_frames
from lc_veh_selection import find_lc_frame

class ngsimDataSelection():

    def __init__(self, data_path= '../../../trajectories-0805am-0820am.csv', 
                 t_h=30, t_f=50, d_s=2, enc_size = 64, grid_size = (13,3)):
        print('\n working on {} \n'.format(data_path))
        self.df_0 = pd.read_csv(data_path)
        self.veh_id2traj_l12345678 = self.get_veh_id2traj_lane12345678()
 
    def get_veh_id2traj_lane12345678(self):
        print('getting trajectories of vehicles...')
        veh_IDs = set(self.df_0['Vehicle_ID'].values)
        veh_id2traj = {}
        for vi in veh_IDs:
            vi_lane_ids = set(self.df_0[(self.df_0['Vehicle_ID']==vi)]['Lane_ID'].values)
            vi_traj =  self.df_0[(self.df_0['Vehicle_ID']==vi)][['Frame_ID', 'Vehicle_ID', 'Local_X', 'Local_Y', 'Lane_ID', 'v_Length', 'v_Width', 'v_Vel', 'v_Acc', 'Preceeding', 'Following', 'Space_Hdwy', 'Time_Hdwy']]
            veh_id2traj[vi] = vi_traj
        print('totally {} vehicles stay in lane 1 to 8 from frame {} to frame {}'.format(len(veh_id2traj), ' ',' ' ))
        return veh_id2traj
    
    def get_frames(self, df0, frm_stpt=12, frm_enpt=610):
        vehposs = df0[ (df0['Frame_ID']>=frm_stpt)
                            &(df0['Frame_ID']<frm_enpt)][['Frame_ID', 'Vehicle_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc']]
        return vehposs

    def get_data_item(self, ego_id, frm_id):
        nb9 = self.get_nbrs9(ego_id, frm_id)
        # print(nb9)

        hist9 = np.zeros([9,31,2])
        f1 = np.zeros([1,50,2])
        if 0 in nb9:
            return np.empty([0, 31, 4]), np.empty([0, 50, 4])
        for i, vi in enumerate(nb9):
            h = self.get_hist(vi, ego_id, frm_id)
            if len(h)==0:
                return np.empty([0, 31, 4]), np.empty([0, 50, 4])
            else:
                hist9[i] = h
        f = self.get_fut(ego_id, ego_id, frm_id)
        
        if len(f)==0:
            return np.empty([0, 31, 4]), np.empty([0, 50, 4])
        f1[0] = f 
        return hist9, f1

    def get_nbrs(self, ego_id, frm_id, lane='left'):
        if lane=='ego':
            tar_pre_vid, tar_fol_vid = self.veh_id2traj_l12345678[ego_id][self.veh_id2traj_l12345678[ego_id]['Frame_ID']==frm_id][['Preceeding', 'Following']].values[0]
            nbrs = [int(tar_fol_vid), int(ego_id), int(tar_pre_vid)]
            return nbrs
        ego_y, ego_lane = self.veh_id2traj_l12345678[ego_id][self.veh_id2traj_l12345678[ego_id]['Frame_ID']==frm_id][['Local_Y', 'Lane_ID']].values[0]
        if lane=='left':
            vehs_tar_lane = self.df_0[(self.df_0['Frame_ID']==frm_id) & (self.df_0['Lane_ID']==int(ego_lane-1))][['Vehicle_ID', 'Local_Y']].values
        if lane=='right':
            vehs_tar_lane = self.df_0[(self.df_0['Frame_ID']==frm_id) & (self.df_0['Lane_ID']==int(ego_lane+1))][['Vehicle_ID', 'Local_Y']].values
        if vehs_tar_lane.size==0:
            return [0, 0, 0]
        vehs_tar_lane[:,1] = abs( vehs_tar_lane[:,1] - ego_y ) # the longitudinal distance
        tar_mid_idx = np.where(vehs_tar_lane[:,1]==np.min(vehs_tar_lane[:,1]))[0][0]
        tar_mid_vid = vehs_tar_lane[:,0][tar_mid_idx]
        tar_pre_vid, tar_fol_vid = self.veh_id2traj_l12345678[tar_mid_vid][self.veh_id2traj_l12345678[tar_mid_vid]['Frame_ID']==frm_id][['Preceeding', 'Following']].values[0]
        nbrs = [ int(tar_fol_vid ), int(tar_mid_vid), int(tar_pre_vid) ]
        return  nbrs

    def get_nbrs9(self, ego_id, frm_id):
        nbrs9 = []
        for l in ['left', 'ego', 'right']:
            nbrs = self.get_nbrs(ego_id, frm_id, lane=l)
            nbrs9+=nbrs
        return nbrs9
    
    def get_hist(self, veh_id, ego_id, frm_id, hist_len=30):
        if veh_id not in self.veh_id2traj_l12345678.keys():
            return np.empty([0,2])
        ref_pos = self.veh_id2traj_l12345678[ego_id][self.veh_id2traj_l12345678[ego_id]['Frame_ID']==frm_id][['Local_X', 'Local_Y']].values[0]
        veh_track = self.get_frames(self.veh_id2traj_l12345678[veh_id], frm_stpt=frm_id-hist_len, frm_enpt=frm_id+1)
        veh_track = veh_track[['Local_X', 'Local_Y']].values
        veh_track = veh_track - ref_pos
        if len(veh_track) < hist_len+1:
            return np.empty([0,2])
        return veh_track

    def get_fut(self, veh_id, ego_id, frm_id, fut_len=50):
        ref_pos = self.veh_id2traj_l12345678[ego_id][self.veh_id2traj_l12345678[ego_id]['Frame_ID']==frm_id][['Local_X', 'Local_Y']].values[0]
        veh_track = self.get_frames(self.veh_id2traj_l12345678[veh_id], frm_stpt=frm_id+1, frm_enpt=frm_id+fut_len+1)
        veh_track = veh_track[['Local_X', 'Local_Y']].values
        veh_track = veh_track - ref_pos
        if len(veh_track) < fut_len:
            return np.empty([0,2])
        return veh_track

def get_data_pieces(traj_time='-0805am-0820am'):
    d_path = '../../trajectories' + traj_time +'.csv'

    dset = ngsimDataSelection(data_path= d_path)

    Hist = np.empty([0,9,31,2]).astype('float16')
    Fut = np.empty([0,1,50,2]).astype('float16')

    count = 0
    bad_count = 0
    if traj_time.split('-')[1] == '0750am':
        vehs_LC_once = vehs_LC_0750
        veh_id2traj = veh_id2traj_0750
    elif traj_time.split('-')[1] == '0805am':
        vehs_LC_once = vehs_LC_0805
        veh_id2traj = veh_id2traj_0805
    elif traj_time.split('-')[1] == '0820am':
        vehs_LC_once = vehs_LC_0820
        veh_id2traj = veh_id2traj_0820
    else:
        print('use correct traj name!')
        
    print(len(vehs_LC_once))
    for vi in vehs_LC_once:
        vi = vi
        print('veh id {}, count {}'.format(vi, count))
        traji = veh_id2traj[vi]
        lc_f,ol,tl = find_lc_frame(traji)
        for fid in range(lc_f-130, lc_f+130):
            try:
                hh, ff = dset.get_data_item(vi, fid)
            except:
                bad_count +=1
                continue
            if hh.shape[0]==0:
                continue
            Hist = np.concatenate((Hist, np.expand_dims(hh, axis=0).astype('float16')))
            Fut = np.concatenate((Fut, np.expand_dims(ff, axis=0).astype('float16')))
            count+=1
    #     break
    print(Hist.shape)
    print(Fut.shape)
    print(count)
    hist_data_name = 'HIST_{}lc_selection'.format(traj_time.split('-')[1])
    fut_data_name = 'FUT_{}lc_selection'.format(traj_time.split('-')[1])
    np.save(hist_data_name, Hist)
    np.save(fut_data_name, Fut)


if __name__ == '__main__':
    # selected vehicles
    vehs_LC_0750 = np.load('VehicleIDwithONElcTraj_selection0750am.npy')
    vehs_LC_0805 = np.load('VehicleIDwithONElcTraj_selection0805am.npy')
    vehs_LC_0820 = np.load('VehicleIDwithONElcTraj_selection0820am.npy')

    # raw NGSIM trajectories
    path_raw_data_0750 = '../../trajectories-0750am-0805am.csv'
    path_raw_data_0805 = '../../trajectories-0805am-0820am.csv'
    path_raw_data_0820 = '../../trajectories-0820am-0835am.csv'

    df_0750 = pd.read_csv(path_raw_data_0750)
    df_0805 = pd.read_csv(path_raw_data_0805)
    df_0820 = pd.read_csv(path_raw_data_0820)

    # vehicle id to trajectories {}
    veh_id2traj_0750 = get_veh_id2traj(df_0750)
    veh_id2traj_0805 = get_veh_id2traj(df_0805)
    veh_id2traj_0820 = get_veh_id2traj(df_0820)

    # print(len(vehs_LC_0750))
    # get data pieces
    get_data_pieces(traj_time='-0750am-0805am')
    get_data_pieces(traj_time='-0805am-0820am')
    get_data_pieces(traj_time='-0820am-0835am')