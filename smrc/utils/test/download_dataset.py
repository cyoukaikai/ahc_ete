# from BeautifulSoup import BeautifulSoup
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re

#https://arstechnica.com
#http://synthia-dataset.net/download-2/
url_str = 'http://synthia-dataset.net/download-2/'
html_page = urlopen(url_str)
soup = BeautifulSoup(html_page)
links = []

for link in soup.findAll('a', attrs={'href': re.compile("^http://")}):
    links.append(link.get('href'))
    print(link.get('href'))

# print(links)

link_filtered = []
for link in links:
    if link.find('http://synthia-dataset.net/download/') > 0:
        link_filtered.append(link)


print(link_filtered)
