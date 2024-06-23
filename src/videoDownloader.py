from pytube import YouTube

class youtubeDownloader:
    def __init__(self, videoDownloadPath='./YouTube-Videos/'):
        self.videoDownloadPath = videoDownloadPath
       
    
    def downloadVideos(self, videoURLs):
        for videoURL in videoURLs:
            yt = YouTube(videoURL)
            stream = yt.streams.get_highest_resolution()
        
            print(f"Downloading video: {yt.title}...")
            stream.download(output_path=self.videoDownloadPath)
            print("Video downloaded successfully!")
    
        
if __name__ == "__main__":
    video_download_directory = './YouTube-Videos/'
    videoURLs = ["https://www.youtube.com/watch?v=WeF4wpw7w9k&t=6s",
        "https://www.youtube.com/watch?v=2NFwY15tRtA",
        "https://www.youtube.com/watch?v=5dRramZVu2Q"] 

    downloader = youtubeDownloader(videoDownloadPath=video_download_directory)
    downloader.downloadVideos(videoURLs)