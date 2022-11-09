import yt_dlp
import json
import tqdm
import os
import shutil
from os import listdir
from os.path import isfile, join
from pathlib import Path
import json, requests

def get_playlist_metadata(playlist_url: str):
  ydl_opts = {
      'ignoreerrors': True,
      'extract_flat': True, 
      'skip_download': True,
      'quiet': True
  }

  with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      info = ydl.extract_info(playlist_url, download=False)
  
  return info

# otehr comment

def get_playlist_items(playlist_id: str, api_key: str):
  URL1 = 'https://www.googleapis.com/youtube/v3/playlistItems?part=contentDetails&maxResults=50&fields=items/contentDetails/videoId,nextPageToken&key={}&playlistId={}&pageToken='.format(api_key, playlist_id)

  next_page = ''
  vid_list = [] 

  while True:
      results = json.loads(requests.get(URL1 + next_page).text)
      
      for x in results['items']:
          vid_list.append('https://www.youtube.com/watch?v=' + x['contentDetails']['videoId'])

      if 'nextPageToken' in results:
          next_page = results['nextPageToken']
      else:
          break

  return vid_list


def process_downloaded_audio(audioFile: str):
    # split file into smaller chunks
    print("  Splitting voice sections ...", end='')
    os.system("python split_segments.py --file_path "+ audioFile +" --cleanup_vocals_files")
    print("  [done] ")

    # upload splits to S3
    audio_file_name = Path(audioFile).stem
    parent_folder = Path(audioFile).parent

    print("  Uploading to S3 ...", end='')

    split_output_folder = join(parent_folder, 'output-8',audio_file_name,"chunks-withMin")
    split_files = [join(split_output_folder, f) for f in listdir(split_output_folder) if (isfile(join(split_output_folder, f)) and f.endswith(".mp3"))]
    for split in split_files:
      split_file_name = os.path.basename(split)
      s3.put(split, f"s3://{s3_root_path}/{s3_output_folder}/{audio_file_name}_{split_file_name}")
    
    print("  [done] ")

    # delete temp files
    print("  Deleting temp files ...", end='')
    os.remove(audioFile)
    shutil.rmtree(join(parent_folder, 'output-8',audio_file_name))
    print("  [done] ")


def download_progress_hook(d):
  if d['status'] == 'finished':
      filename=d['filename']
      print(" "+filename)
      process_downloaded_audio(os.path.abspath(d['filename']))

def download_playlist_items(playlistInfo, videoUrls):
    playlistTitle = playlistInfo['title'].replace(' ', '_').lower()
    playlistId = playlistInfo['id']

    # dowload options
    downloaded_output_template = f"dataset/{playlistTitle}_%(upload_date>%Y-%m-%d)s_%(id)s.wav"
    ydl_opts = {
        'format': 'bestaudio/best',
        "audio-format": "wav",
        'outtmpl': downloaded_output_template,
        'ignoreerrors': True,
        'no-playlist': True,
        'quiet': True,
        'cookies': '/content/youtube.com_cookies.txt',
        'progress_hooks': [download_progress_hook]
    }
    workingFolder = 'dataset'

    #S3 metadata

    progress_tracker = f"progress_{playlistTitle}_{playlistId}.log"
    progress_tracker_s3 = f"s3://{s3_root_path}/{progress_tracker}"
    if(s3.exists(progress_tracker_s3)):
        s3.download(progress_tracker_s3,progress_tracker)
    else:
        # create empty file
        with open(progress_tracker, 'w') as fp:
            pass

    with open(progress_tracker) as file:
        processed_lines = [line.rstrip() for line in file]

    for record in tqdm.notebook.tqdm(videoUrls, desc="Downloading"):
        try:
          current_url = record
        except:
          continue
        if (current_url in processed_lines):
          print("skipping "+ current_url)
          continue

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download(current_url)

        # Append current URl to end of file
        with open(progress_tracker, "a") as file_object:
          file_object.write(current_url)
          file_object.write("\n")
        
        # save file
        s3.put(progress_tracker, progress_tracker_s3)