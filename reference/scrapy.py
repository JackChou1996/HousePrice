from bs4 import BeautifulSoup
import requests


url = "https://cn.tripadvisor.com/Attraction_Review-g60763-Activities-New_York_City_New_York.html"
wb_data = requests.get(url)

soup  = BeautifulSoup(wb_data.text,'lxml')
print(soup)
#titles = soup.select("#lithium-root > main > div._2zSZ5If6 > div > div > div > div:nth-child(5) > div._1h6gevVw > div:nth-child(1) > div > div._1fqdhjoD > a._255i5rcQ")
titles = soup.select('div._1fqdhjoD > a[class = "_1r6cJ4GV"]')
print(titles)
titles[0].get_text()
cates = soup.select('span[class = "_21qUqkJx"]')
print(cates)
list(cates[0].stripped_strings)
<span class="_21qUqkJx">公园&amp;自然景点</span>
imgs = soup.select('div[ class = "ZVAUHZqh"]')
print(imgs)
<div class="_2oCMdf3A"><div class="ZVAUHZqh" style="background-image: url(&quot;https://dynamic-media-cdn.tripadvisor.com/media/photo-o/08/f2/87/f7/the-high-line.jpg?w=300&amp;h=200&amp;s=1&quot;); background-size: cover; height: 100%; width: 100%;"></div></div>

for title,img,cate in zip(titles,imgs,cates):
    data = {
        'title':title.get_text(),
        'img':img.get('src'),
        'cate':list(cate.stripped_strings)
    }
    print(data)

#mongodb
import pymango
client = pymango.MongoClient('localhost',27817)
walden = client('walden')
sheet_tab = walden('sheet_tab')

path = r'C:\Users\Jack\Desktop\zoom.txt'
with open(path,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for index,line in enumerate(lines):
        data = {
            'index':index,
            'line':line,
            'words':len(line.split())
        }
        print(data)
        sheet_tab.insert_one(data)

for item in sheet_tab.find({'words':{'$lt':5}}):
    print(item)








url_saves = ''