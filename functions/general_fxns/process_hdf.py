import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import glob
import tables
import pickle
import numpy as np
import pandas as pd


def process_hdf(path, mode, CONTROL=False, index=0):

    os.chdir(path)

    decoder_files = glob.glob('*.pkl')
    
    if CONTROL==False:
        
        if mode == 'rotation':
            hdf = tables.open_file(glob.glob('*.hdf')[0])
    
        elif mode == 'shuffle':
            hdf = tables.open_file(glob.glob('*.hdf')[1])
    
    else:
        
        hdf = tables.open_file(glob.glob('*.hdf')[index])
        
        


    KG_picklename = decoder_files[1]
        

    '''Load KG values for task.'''
    f = open(KG_picklename, 'rb')
    KG_stored = []
    while True:
        try:
            KG_stored.append(pickle.load(f))
        except Exception:
            break
    f.close()
    
    KG_stored = np.array(KG_stored)
    
    xBL = KG_stored[1000,3,:]
    yBL = KG_stored[1000,5,:]
    xPE = KG_stored[15000,3,:]
    yPE = KG_stored[15000,5,:]
    
    shuffledX = xBL!=xPE
    shuffledY = yBL!=yPE
    
    if np.sum(shuffledX != shuffledY) >= 1:
        print('ERROR IN KG')
    
    
    dfKG = pd.DataFrame({'shuffled': shuffledX, 'xBL': xBL, 'yBL': yBL, 'xPE': xPE, 'yPE': yPE})
    
    
    '''Parse trial task messages from HDF.'''
    msg = hdf.root.task_msgs[:]['msg'] 
    ind = hdf.root.task_msgs[:]['time']
    
    pert = hdf.root.task[:]['pert'][:,0]
    error_clamp = hdf.root.task[:]['error_clamp'][:,0]
    block_type = hdf.root.task[:]['block_type'][:,0]
    
    reward_msg = np.where(msg == b'reward')[0]
    
    start_moving_msg = np.subtract(reward_msg,3)
    stop_moving_msg  = np.subtract(reward_msg,2)
    end_hold_msg     = np.subtract(reward_msg,1)
    
    start_moving_ind = ind[start_moving_msg]
    stop_moving_ind  = ind[stop_moving_msg]
    end_hold_ind     = ind[end_hold_msg]
    
    
    df = pd.DataFrame({'start':      start_moving_ind,
                        'stop':       stop_moving_ind, 
                        'end_hold':   end_hold_ind,
                        'pert':       pert[stop_moving_ind],
                        'errorClamp': error_clamp[stop_moving_ind],
                        'blockType':  block_type[stop_moving_ind]})
    
    
    '###################################'
    '''   Determine Target Locations '''
    '###################################'
    
    fix = lambda r:r+(2*np.pi) if r < 0 else r #lambda d:d+360 if d < 0 else d
    
    target_xloc = hdf.root.task[:]['target'][:,0]
    target_yloc = hdf.root.task[:]['target'][:,2]
    target_loc_rads = np.array([fix(np.arctan2(y,x)) for y,x in zip(target_xloc, target_yloc)])
    target_loc_degs = (np.array(target_loc_rads)*(180/np.pi)).astype(int)
    
    df['target_rads'] = target_loc_rads[stop_moving_ind]
    df['target']      = target_loc_degs[stop_moving_ind]

    
    '####################################'
    ''' Cursor Kinematics & BMI Update '''
    '####################################'
    
    dfCursor = pd.DataFrame({'cursor_px': hdf.root.task[:]['cursor'][:,0], 
                              'cursor_py': hdf.root.task[:]['cursor'][:,2],
                              'decoder_px': hdf.root.task[:]['decoder_state'][:,0,0],
                              'decoder_py': hdf.root.task[:]['decoder_state'][:,2,0],
                              'decoder_vx': hdf.root.task[:]['decoder_state'][:,3,0],
                              'decoder_vy': hdf.root.task[:]['decoder_state'][:,5,0],
                              })
    
    
    
    dfCursor['update'] = np.sum(( np.abs(dfCursor.diff(axis=0)) > 0).astype(int), axis=1).values


    '####################################'
    ''' Metrics '''
    
    """
    trial time (s)
    cursor path length (cm)
    spiking data
    """
    
    '####################################'
    
    'Trial Time (s)'
    df['trial_time'] = ((df['stop'] - df['start'])+1)/60 #60 = 60Hz hdf update rate
    
    
        
    'Total Distance Cursor Travels (i.e., path length)'

    cursor_px = []
    cursor_py = []
    
    decoder_px = []
    decoder_py = []
    decoder_vx = []
    decoder_vy = []
    
    cursor_distance  = []
    decoder_distance = []
    
    spikes = hdf.root.task[:]['spike_counts'][:,:,0] 

    trial_spike_counts = []
    

    for test, t, i, j in zip(np.arange(len(df)), df['trial_time'], df['start'].values, df['stop'].values):
    
        cursor_px_ = dfCursor['cursor_px'][i:j].values
        cursor_py_ = dfCursor['cursor_py'][i:j].values
        
        decoder_px_ = dfCursor['decoder_px'][i:j].values
        decoder_py_ = dfCursor['decoder_py'][i:j].values
        decoder_vx_ = dfCursor['decoder_vx'][i:j].values
        decoder_vy_ = dfCursor['decoder_vy'][i:j].values
       
        update_inds = np.where(dfCursor['update'][i:j].values > 0)[0]   
        
        cursor_px.append(cursor_px_[update_inds])
        cursor_py.append(cursor_py_[update_inds])
        
        decoder_px.append(decoder_px_[update_inds])
        decoder_py.append(decoder_py_[update_inds])
        
        decoder_vx.append(decoder_vx_[update_inds])
        decoder_vy.append(decoder_vy_[update_inds])
        
        cursor_distance.append(  np.linalg.norm(np.vstack((cursor_px_[update_inds], cursor_py_[update_inds]))) )
        decoder_distance.append(  np.linalg.norm(np.vstack((decoder_px_[update_inds], decoder_py_[update_inds]))) )
    
        spike_update_inds = np.arange(i,j)[[update_inds]]
        sc = []
        for ui in spike_update_inds: 

            sc.append( np.sum(spikes[ui-5:ui+1, :], axis=0) )
        
        trial_spike_counts.append(sc)
    
    df['cursor_px'] = cursor_px
    df['cursor_py'] = cursor_py
    
    df['decoder_px'] = decoder_px
    df['decoder_py'] = decoder_py
    df['decoder_vx'] = decoder_vx
    df['decoder_vy'] = decoder_vy
    
    df['cursor_distance'] = cursor_distance
    df['decoder_distance'] = decoder_distance
    
    df['spikes'] = trial_spike_counts
    
    return(df, dfKG) 



    

