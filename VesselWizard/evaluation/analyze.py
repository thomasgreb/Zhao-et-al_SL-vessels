import glob, sys, os, json, pandas, cv2, numpy as np

data = []

for file in glob.iglob(os.path.join(sys.argv[1],r"**\settings.json"), recursive=True, include_hidden=True):
    directory = os.path.basename(os.path.dirname(file))
    dirs = os.path.dirname(file)
    if not directory.startswith(".metadata-"):
        continue
    try:
        fijiresult = glob.glob(os.path.join(dirs,r"result*fiji.tif*"))+glob.glob(os.path.join(dirs,r"result*fiji.png*"))
        if len(fijiresult) > 0:
            img = cv2.imread(fijiresult[0])
            print("Found a fiji result")
        else:
            img = cv2.imread(os.path.join(dirs,"result.png"))
            print("Found no fiji", fijiresult)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0,230,230])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
        lower_red = np.array([170,230,230])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
        mask = mask0+mask1
        cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (x,y),radius = cv2.minEnclosingCircle(np.vstack(cnt))
        enclosed = cv2.convexHull(np.vstack(cnt))
        enclosed = np.array([px[0] for px in enclosed])
        hull = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(hull, [enclosed], -1, (1,), cv2.FILLED)
        hullsum = np.sum(hull)
        img2 = img.copy()
        cv2.circle(img2, (round(x),round(y)), round(radius), (0,255,0),10)
        cv2.drawContours(img, [enclosed], -1, (0,255,0), thickness=10)
        cv2.imwrite(os.path.join(dirs,"result_with_xylem_area.png"),img)
        cv2.imwrite(os.path.join(dirs,"result_with_xylem_circle.png"),img2)

        fijiarea = 0
        fijicount = 0
        fijifile = glob.glob(os.path.join(dirs,r"*.csv"))
        if len(fijifile) > 0:
            print("found fijicsv")
            with open(fijifile[0],"r") as filex:
                area = None
                for line in filex:
                    fline = line.split(",")
                    if area is None:
                        area = fline.index("Area")
                    else:
                        fijicount+=1
                        fijiarea+=int(fline[area])
        else:
            print("found no fijicsv")


    except Exception as e:
        print(e)
        radius = 0
        hullsum = 0
        fijiarea = 0
        fijicount = 0

    with open(file,"r") as p:
        s = json.load(p)

    data.append({
        "dirs": dirs,
        "directory": directory[10:],
        "color": "red",
        "x": s["size"]["x"],
        "y": s["size"]["y"],
        "xorg": s["size"]["xorg"],
        "yorg": s["size"]["yorg"],
        "radius": radius,
        "cellcount": s["statistics"]["countMatched"],
        "areapx": s["statistics"]["areaMatched"],
        "circleareapx": s["statistics"]["circleArea"],
        "pixelsize": s["statistics"]["pixelsize"],
        "convexhull": hullsum,
        "fijiarea": fijiarea,
        "fijicount": fijicount
    })

    if "countMatched2" in s["statistics"] and s["statistics"]["countMatched2"] > 0:
        data.append({
            "dirs": dirs, 
            "directory": directory[10:],
            "color": "green",
            "x": s["size"]["x"],
            "y": s["size"]["y"],
            "xorg": s["size"]["xorg"],
            "yorg": s["size"]["yorg"],
            "cellcount": s["statistics"]["countMatched2"],
            "areapx": s["statistics"]["areaMatched2"],
            "circleareapx": s["statistics"]["circleArea"],
            "pixelsize": s["statistics"]["pixelsize"],
            "convexhull": 0
        })

df = pandas.DataFrame(data)
df.to_excel(os.path.join(sys.argv[1],"analyse.xlsx"), sheet_name="Analyse")