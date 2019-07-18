
import numpy as np
import pandas as pd
import instaloader
import tqdm

#! pip install instaloader

data = pd.DataFrame(columns=['username','image_url', 'following', 'followers', 'date', 'time', 'likes', 'caption'])
info = []

def scraper(hashtag):
  L = instaloader.Instaloader()
  posts = L.get_hashtag_posts(hashtag)
  
  likes = []
  caption = []
  date = []
  time = []
  image_url = []
  username = []
  following = []
  followers = []
  count = 0

  for i in tqdm(posts):
    if i.is_video == False:
      likes.append(i.likes)
      k = str(i.caption)
      k = k.replace('\n', " ")
      caption.append(k)
      date.append(i.date.strftime("%d-%m-%Y"))
      time.append(i.date.strftime("%H:%M:%S"))
      image_url.append(i.url)
      profile = instaloader.Profile.from_username(L.context, i.owner_username)
      username.append(i.owner_username)
      following.append(profile.followees)
      followers.append(profile.followers)
      count+=1
      if count == 1000 :
        break
        
  
  user_data = pd.DataFrame(list(zip(username, image_url, following, followers, date, time, likes, caption)),
              columns=['username','image_url', 'following', 'followers', 'date', 'time', 'likes', 'caption'])
  
  global data
  data = data.append(user_data, ignore_index=True, sort=False)
  print(data.shape)
  
  info.append([hashtag, len(likes)])
  print(info)


scraper('modichod')

data

data.to_excel('data.xlsx', index=False)
