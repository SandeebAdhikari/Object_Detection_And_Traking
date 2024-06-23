from pytube import YouTube
from fastapi import FastAPI
from typing import List

app = FastAPI()
class youtubeDownloader:
    def __init__(self, videoDownloadPath='./Video_Search_Assignment/Downloads/YouTube-Videos/',captionDownloadPath ='./Video_Search_Assignment/Downloads/Captions/'):
        self.videoDownloadPath = videoDownloadPath
        self.captionDownloadPath = captionDownloadPath
    
    def downloadVideoWithCaptions(self, videoURLs):
        results =[]
        for videoURL in videoURLs:
            try:
                yt = YouTube(videoURL)
                stream = yt.streams.get_highest_resolution()
            
                print(f"Downloading video: {yt.title}...")
                stream.download(output_path=self.videoDownloadPath)
                video_status ="Video downloaded successfully!"
                 

                if yt.captions:
                    caption = yt.captions.get_by_language_code('a.en')
                    if caption:
                        print("Downloading captions...")
                        caption.download(title= yt.title, srt= False ,output_path=self.captionDownloadPath)
                        caption_status ="Captions downloaded successfully!" 
                    else:
                        caption_status = "No English caption available for this video."
                results.append({"title": yt.title, "video_status": video_status, "caption_status": caption_status})
            except Exception as e:
                results.append({"url": videoURL, "error": str(e)})
        return results

@app.post("/download-videos/")
async def download_videos(videoURLs: List[str]):
    downloader = youtubeDownloader()
    results = downloader.downloadVideoWithCaptions(videoURLs)
    return results