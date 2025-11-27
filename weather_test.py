import requests
import tkinter as tk

api_key = "SWOiSWkZWBrzXEjHx"#改为自己的api
city_id = "苏州"

# 查询天气信息
url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={city_id}&language=zh-Hans&unit=c&fields=location,update_time"
response = requests.get(url)
data = response.json()
print(data)


# 检查是否成功获取数据
if "results" in data:
    weather = data["results"][0]["now"]["text"]
    temperature = data["results"][0]["now"]["temperature"]
    location = data["results"][0]["location"]["name"]
    update_time = data["results"][0]["last_update"]

    # 创建窗口
    root = tk.Tk()
    root.title("Weather Information")

    # 显示天气信息
    label_weather = tk.Label(root, text=f"Weather: {weather}")
    label_weather.pack()

    label_temperature = tk.Label(root, text=f"Temperature: {temperature}°C")
    label_temperature.pack()

    label_location = tk.Label(root, text=f"Location: {location}")
    label_location.pack()

    label_update_time = tk.Label(root, text=f"Last Update: {update_time}")
    label_update_time.pack()

    root.mainloop()
else:
    print("Error: Unable to retrieve weather information.")