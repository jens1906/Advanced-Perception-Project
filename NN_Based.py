from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="SoccerNet")
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])
