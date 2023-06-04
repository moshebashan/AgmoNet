# -*- coding: utf-8 -*-

"""
Function: read_labels
    
Description:
This function splits long recordings into short labeled segments (samples)
based on specified labels in text files. It takes as input WAV files and their
corresponding text files containing the labels. The text files should have the
same name as the corresponding WAV files.
This function is compatible with Audacity software, with or without
spectral selection (with minimum and maximum frequency).
   
Input Parameters:
- milon_path: Path to the MILON file or a DataFrame/text/CSV file containing all classes (species) names, IDs, and labels.
- files_path: Path to the directory containing the WAV files and their corresponding text files.
- save_files: A boolean argument. If set to True, it will save all the signals in the 'results' folder.
   
Return Value:
- samples: A DataFrame containing all the signals (optional to use).
  
Example Usage:
milon = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\python\readl" 
files_path = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\Recordings (new)\recording collections\XC\חרטומית\מתויג" 
save_files = True # It will save the signals in the 'results' folder
samples = read_labels(milon_path, files_path, save_files)  # It will output the signals in a DataFrame
  
"""

# IMPORT PACKAGES:
import librosa, librosa.display
import glob, os
import numpy as np
import pandas as pd
import soundfile as sf

def read_labels(milon,files_path,save_files):
    if isinstance(milon, str):
        os.chdir(milon)
        milon = pd.read_csv("milon.txt", sep="\t", header=None, encoding=('iso8859_8')) # read dataframe with all species nams, id's and labels        
    os.chdir(files_path)
    if not os.path.isdir('results'):
        os.mkdir('results') # make new folder for results    
    wav_files = glob.glob("*.wav") # make list of all wav files
    text_files = glob.glob("*.txt") # make list of all text files
    
    samples = pd.DataFrame({'file name': [],'signal': []})
    unidentified = pd.DataFrame(columns=('file name', 'index', 'lable')) # new dataframe for unidentified labels 
    unidentified_row = 0
        
    for i in range(len(wav_files)): # loop over all files
        file_name = wav_files[i] # get the file name
        fname = file_name[0:len(file_name)-4] # get the file name without the suffix
        #signal, sr = librosa.load(file_name, sr=22050) # read the wav file
        text_file = pd.read_csv(fname + '.txt', sep="\t", header=None) # read the text file       
        if text_file.shape[0]<=1 or text_file[0][1] != '\\': # if spectral selection mode is disable or enable
            spectral_selection = 'disable'
            labels = text_file[2]
            time_start = text_file[0] # time start in sec
            time_end = text_file[1] # time end in sec
            labels_df = pd.concat([labels.reset_index(drop=True), time_start.reset_index(drop=True), 
                               time_end.reset_index(drop=True)], axis = 1)
            labels_df.columns = ['labels','time start','time end']
        else:
            spectral_selection = 'enable'
            labels = text_file[2][0::2].to_frame() # get all lables from the text file
            time_start = text_file[0][0::2].to_frame() # get the start time from all samples
            time_end = text_file[1][0::2].to_frame() # get the end time from all samples
            freq_min = text_file[1][1::2].to_frame() # get the minimum frequency from all samples
            freq_max = text_file[2][1::2].to_frame() # get the maximum frequency from all samples
            labels_df = pd.concat([labels.reset_index(drop=True), time_start.reset_index(drop=True), 
                               time_end.reset_index(drop=True), freq_min.reset_index(drop=True), freq_max.reset_index(drop=True)], axis = 1)
            labels_df.columns = ['labels','time start','time end','freq min','freq max']
               
        for j in range(len(labels_df)): # loop over all lables in the file
            index = j+1 # index number of sample in the recording
            label = labels_df['labels'][j] # lable name (species)
            col = 0
            match = milon[milon[col] == label] # find match for lable from file with label table (in column=1)
            while col < np.shape(milon)[1]-1:  # look for label match in different column if label is unidentified
                col = col + 1
                if match.shape[0] == 0:
                    match = milon[milon[col] == label] # find match for lable from file with label table (in column=col)
            
            if match.shape[0] == 0: # if label is unidentified
                unidentified = True
                unidentified.loc[unidentified_row] = [fname, index, label] # update unidentified table
                unidentified_row = unidentified_row + 1
            elif match.shape[0] != 0: # if lable is identified:
                species_id = int(match[0]) # in file index lable
                new_file_name = str(species_id) + '_' + str(index) + '_' + file_name # new file name             
                samp_start = float(labels_df['time start'][j]) # start time of the sample in seconds
                samp_duration = float(labels_df['time end'][j]) - float(labels_df['time start'][j]) # duration of the sample in seconds
                sample, sr = librosa.load(wav_files[i], offset = samp_start, duration = samp_duration, sr = 44100) # read the audio signal                
                if sr != 44100: # resample to 44.100 kHz if otherwise 
                    librosa.resample(sample, sr, 44100)               
                # write the sample in the results DataFrame and folder :
                samples.loc[len(samples)] = (new_file_name,sample)
                if save_files == True:                    
                    os.chdir('results') # change to results folder
                    sf.write(new_file_name,sample, samplerate = sr) # write the wav file
                    os.chdir(files_path) # change back to working directory
                
                # update log:
                labels_log = pd.DataFrame(columns=('file name', 'label', 'index', 'start time', 'end time', 'min freq', 'max freq')) # new dataframe for lables log
                labels_log_row = 0
                                          
                if spectral_selection == "enable": # update metadata with or without spectral selection
                    labels_log.loc[labels_log_row] = [fname,labels_df['labels'][j],index,labels_df['time start'][j],labels_df['time end'][j],labels_df['freq min'][j],labels_df['freq max'][j]]
                    labels_log_row = labels_log_row + 1
                else:
                    labels_log.loc[labels_log_row] = [fname,labels_df['labels'][j],index,labels_df['time start'][j],labels_df['time end'][j],[],[]]
                    labels_log_row = labels_log_row + 1
    
    # save metadata to a pickle file:
    #os.chdir(results_dir)
    #meta.to_pickle("metadata.pkl")   
           
    if not unidentified.empty:
        print("=====  unidentified labels  =====")
        print(unidentified)
        print("=================================")
    return samples   
    print("======      End Of Process      ======")



milon = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\python\readl" # label table path
files_path = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\Recordings (new)\recording collections\XC\חרטומית\מתויג" # set to current folder
save_files = True
samples = read_labels(milon, files_path, save_files)
#print(head(signals)) 
 
##################################
'''
לכתוב הערות
לכתוב הסבר מפורט בראש הסקריפט
 לסדר תאימות לתוכנות raven ו- avisoft ועוד אם צריך
'''