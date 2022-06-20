import csv
friendInfo=["潘博","张飞","李宁"]
with open(r"friendInformation.csv", 'w+', newline='',encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(friendInfo)