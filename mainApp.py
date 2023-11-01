# Import Libraries #
import cv2
import numpy as np
from tkinter import *
from tkinter import (filedialog, messagebox)  
from tkinter.messagebox import askyesno
from PIL import (Image, ImageTk)
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Functions #
def rgb2gray(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img

def rgb2bnw(image):
    if (len(image.shape) == 3):
        (thresh, img) = cv2.threshold(rgb2gray(image), 127, 255, cv2.THRESH_BINARY)
    else :
        (thresh, img) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return img

def addPixel(image, r, g, b):
    img = image.copy()
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            if (len(img.shape) == 3):
                img[i,j,0] += r  
                img[i,j,1] += g 
                img[i,j,2] += b 
            else:
                img[i,j] += (r+g+b)/3
    return img  

def minPixel(image, r, g, b):
    img = image.copy()
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            if (len(img.shape) == 3):
                img[i,j,0] -= r  
                img[i,j,1] -= g 
                img[i,j,2] -= b 
            else:
                img[i,j] -= (r+g+b)/3
    return img 

def multPixel(image, r, g, b):
    img = image.copy()
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            if (len(img.shape) == 3):
                img[i,j,0] *= r  
                img[i,j,1] *= g 
                img[i,j,2] *= b 
            else:
                img[i,j] *= (r+g+b)/3
    return img 

def divPixel(image, r, g, b):
    img = image.copy()
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            if (len(img.shape) == 3):
                img[i,j,0] /= r  
                img[i,j,1] /= g 
                img[i,j,2] /= b 
            else:
                img[i,j] /= (r+g+b)/3
    return img 

def rgbForm(root, fields):
    entries = {}
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=5, text=field, anchor='w')
        ent = Entry(row)
        ent.insert(0, "0")
        row.pack(side="top", fill=X, padx=5, pady=1)
        lab.pack(side="left")
        ent.pack(side="right", expand=YES, fill=X, padx=1)
        entries[field] = ent
    return entries

def rgbInput(option):
    root = Tk()
    root.title("Input")

    if option == 11:
        header = Label(root, pady=5, text="Addition")
    elif option == 12:
        header = Label(root, pady=5, text="Substraction")
    elif option == 13:
        header = Label(root, pady=5, text="Multiplication")
    elif option == 14:
        header = Label(root, pady=5, text="Division")
    
    button = Button(root, text='Submit', command=lambda: root.quit())
    header.pack(side="top")
    button.pack(side="bottom", padx=5, pady=5)
    entries = rgbForm(root, fields)
    root.after(45000, root.destroy)
    root.mainloop()
    
    return entries

def valInput(option):
    root = Tk()
    root.title("Input")
    row = Frame(root)

    if option == 2:
        header = Label(root, pady=5, text="Sampling")
        label = Label(row, text="Rate",)
    elif option == 3:
        header = Label(root, pady=5, text="Quantization")
        label = Label(row, text="Level")
    elif option == 41:
        header = Label(root, pady=5, text="Increase Intensity")
        label = Label(row, text="Value")
    elif option == 42:
        header = Label(root, pady=5, text="Decrease Intensity")
        label = Label(row, text="Value")
    
    entry = Entry(row)
    entry.insert(0, "0")
    button = Button(root, text='Submit', command=lambda: root.quit()) 

    header.pack(side="top")
    button.pack(side="bottom", pady=5)  
    row.pack(side="bottom", fill=X, padx=5, pady=5)
    label.pack(side="left", padx=1, pady=1)
    entry.pack(side="right", expand=YES, fill=X, padx=1)  

    root.after(45000, root.destroy)
    root.mainloop()

    return int(entry.get())

def sampling(image, rate):
    height, width = image.shape[0], image.shape[1]
    numHeight, numWidth = height/rate, width/rate
    img = np.zeros((height, width, 3), np.uint8)

    for i in range(rate):
        y = int(i*numHeight)
        for j in range(rate):
            x = int(j*numWidth)
            b = image[y, x][0]
            g = image[y, x][1]
            r = image[y, x][2]
            for n in range(int(numHeight)):
                for m in range(int(numWidth)):
                    img[y+n, x+m][0] = np.uint8(b)
                    img[y+n, x+m][1] = np.uint8(g)
                    img[y+n, x+m][2] = np.uint8(r)
    return img

def quantization(image, k):
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret, label, center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    img = center[label.flatten()]
    img = img.reshape(image.shape)
    return img

def incrIntensity(image, x):
    img = image.copy()
    max = img.max()
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            if (len(img.shape) == 3):
                for idx in range(3):
                    if ((img[i,j,idx] + x) >= max):
                        img[i,j,idx] = max
                    else: 
                        img[i,j,idx] += x 
            else :
                if ((img[i,j] + x) >= max):
                    img[i,j] = max
                else: 
                    img[i,j] += x
    return img

def incrIntensity2(image, value):
    img = image.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def decrIntensity2(image, value):
    img = image.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def decrIntensity(image, x):
    img = image.copy()
    min = img.min()
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            if (len(img.shape) == 3):
                for idx in range(3):
                    if ((img[i,j,idx] - x) <= min):
                        img[i,j,idx] = min
                    else: 
                        img[i,j,idx] -= x  
            else :
                if ((img[i,j] - x) <= min):
                    img[i,j] = min
                else: 
                    img[i,j] -= x
    return img 

def inverse(image):
    img = image.copy()
    max = img.max()
    for i in range(img.shape[0]-1) :
        for j in range(img.shape[1]-1) :
            if (len(img.shape) == 3) :
                for idx in range(3):
                    img[i,j,idx] = max - img[i,j,idx]
            else :
                img[i,j] = max-img[i,j]
    return img  

def inverse2(image):
    img = image.copy()
    return ~img 

def histEqualization(image):
    img = image.copy()
    if (len(img.shape) == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img[:,:,0] = cv2.equalizeHist(img[:,:,0])
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    else :
        img =  cv2.equalizeHist(img)
    return img

def histMatching(image, refer):
    img = image.ravel()
    ref = refer.ravel()

    o_values, bin_idx, o_counts = np.unique(img, return_inverse=True,return_counts=True)
    b_values, b_counts = np.unique(ref, return_counts=True)

    o_quantiles = np.cumsum(o_counts).astype(np.float64)
    o_quantiles /= o_quantiles[-1]
    b_quantiles = np.cumsum(b_counts).astype(np.float64)
    b_quantiles /= b_quantiles[-1]

    interp_t_values = np.interp(o_quantiles, b_quantiles, b_values)

    return interp_t_values[bin_idx].reshape(image.shape)

def histSpecification(image, refer):
    img = np.copy(image)
    for i  in range (3) :
        img[:,:,i] = histMatching(image[:,:,i], refer[:,:,i])
    return img

def convFilter(kernel, image):
    img = image.copy()
    kernel = sum(kernel)
    return cv2.filter2D(img,-1,kernel)

def makeHistogram(image):
    if imgStats[0]:
        fig = Figure(figsize=(4.7,4.7), dpi=100)
        plot1 = fig.add_subplot(111)

        if (len(image.shape) == 3) :
            color = ('r','g','b')
            for i, col in enumerate(color):
                histr = cv2.calcHist([image],[i],None,[256],[0,256])
                plot1.plot(histr,color=col)
        else :
            histr = cv2.calcHist([image],[0],None,[256],[0,256])
            plot1.plot(histr)
        
        canvas = FigureCanvasTkAgg(fig, master = tkRoot)
        canvas.draw()

        lst = list(fig.canvas.get_width_height())
        lst.append(3)

        return ImageTk.PhotoImage(Image.fromarray(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(lst)))
    else:
        messagebox.showwarning("Warning", "Open file first!")

def resizeImage(image, new_h, new_w):
    dim = (new_w, new_h)
    img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return img

def changeMod(option):
    global viewMode
    global listLabel
    global listImage
    global imgStats
    global actName
    
    if imgStats[0]:
        closeElement(3)
        listImage[4] = makeHistogram(listImage[0]) 

        if imgStats[1]:
            closeElement(2)
            listImage[4] = makeHistogram(listImage[1]) 
        else:
            listImage[3] = listImage[2]
            actName = "Original"

        if option == 1:
            viewMode = 1
            listLabel[0] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[2])
            listLabel[2] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[3])
        elif option == 2:
            viewMode = 2
            listLabel[0] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[3])
            listLabel[2] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[4])
        
        imgStats[1] = True
        listLabel[0].pack(side="left", padx=10)
        listLabel[2].pack(side="right", padx=10)
        listLabel[1] = Label(tkFrame[0], padx=10, borderwidth=3, relief="ridge", width=101, text=actName, font="Helvetica 12 bold")
        listLabel[1].pack(side="top")
        tkFrame[0].pack()
        tkFrame[1].pack(side="bottom", pady=10)
    else:
        messagebox.showwarning("Warning", "Open file first!")

def showLicense():
    root = Tk()
    root.title("License")

    f = open("LICENSE.md", "r")  
    label = Label(root, text=f.read(), font="Calibri 11")
    label.pack(side="left", padx=10, pady=10)

    root.after(45000, root.destroy)
    root.mainloop()

def openFile():
    global imgPath
    global listLabel
    global listImage
    global imgStats
    global tkFrame

    if imgStats[0]:
        closeElement(1)

    if not imgStats[0]:
        tkFrame[0] = Frame(tkRoot)
        tkFrame[1] = Frame(tkFrame[0])

        imgPath = filedialog.askopenfilename(filetypes=[("JPEG File", "*.jpeg *.jpg"),("JPEG-2000", "*.jp2 .jpx"),("TIFF", "*.tiff"),("SVG", "*.svg"),("GIF", "*.gif"),("BMP", "*.bmp"), ("PNG", "*.png")])
        listImage[0] = np.asarray(Image.open(imgPath))
        listImage[1] = listImage[0].copy()
        
        if (listImage[1].shape[0] >= listImage[1].shape[1]):
            listImage[1] = resizeImage(listImage[1], 400, int((listImage[1].shape[1]*400)/listImage[1].shape[0]))
        else :
            listImage[1] = resizeImage(listImage[1], int((listImage[1].shape[0]*400)/listImage[1].shape[1]), 400)
        
        listImage[2] = ImageTk.PhotoImage(Image.fromarray(listImage[1]))
        listLabel[0] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[2])
        listLabel[0].pack(side="left", padx=10)
        tkFrame[0].pack()
        tkFrame[1].pack(pady=10)
        imgStats[0] = True

def saveFile():
    global listImage

    if imgStats[0]:
        if imgStats[1]:
            listImage[1] = resizeImage(listImage[1], int(listImage[0].shape[0]), int(listImage[0].shape[1])) 
            a = listImage[3].open_path = filedialog.asksaveasfile(initialfile="result.jpg", defaultextension="*.jpg", filetypes=[("JPEG File", "*.jpeg *.jpg")])
            Image.fromarray(listImage[1]).save(a)
        else:
            messagebox.showwarning("Warning", "Give action to image!")
    else:
        messagebox.showwarning("Warning", "Open file first!")

def saveAsFile():
    global listImage

    if imgStats[0]:
        if imgStats[1]:
            listImage[1] = resizeImage(listImage[1], int(listImage[0].shape[0]), int(listImage[0].shape[1])) 
            a = listImage[3].open_path = filedialog.asksaveasfile(initialfile="result.jpg", defaultextension="*.jpg", filetypes=[("JPEG File", "*.jpeg *.jpg"),("JPEG-2000", "*.jp2 .jpx"),("TIFF", "*.tiff"),("SVG", "*.svg"),("GIF", "*.gif"),("BMP", "*.bmp"), ("PNG", "*.png")]) 
            Image.fromarray(listImage[1]).save(a)
        else:
            messagebox.showwarning("Warning", "Give action to image opened first!")
    else:
        messagebox.showwarning("Warning", "Open file first!")

def closeElement(option):
    global imgStats

    if option == 1:
        if imgStats[0]:
            ans = askyesno("Close Confirmation", "Are you sure that you want to close file?")
            if ans:
                imgStats[0] = False
                listLabel[0].destroy()
                tkFrame[0].destroy()
                closeElement(2)
    elif option == 2:
        if imgStats[1]:
            imgStats[1] = False
            listLabel[2].destroy()
            listLabel[1].destroy()
    if option == 3:
        if imgStats[0]:
            listLabel[0].destroy()

def exitApp(root):
    ans = askyesno("Exit Confirmation", "Are you sure that you want to exit application?")
    if ans: root.destroy()

def createRoot(title, icon):
    root = Tk()
    root.title(title)
    root.iconbitmap(icon)

    header = Label(root, padx=0, pady=25, text="ImagiNation", font="Helvetica 20 bold italic")
    header.pack(side="top")

    footer = Label(root, padx=30, pady=10, text="Copyright © mrizkymaulidhan 2021 - All Rights Reserved", font="Calibri 12")
    footer.pack(side="bottom")

    return root

def convertAction(option):
    global imgPath
    global actName
    global listLabel
    global listImage
    global imgStats
    
    if imgStats[0]:
        listImage[1] = listImage[0]

        if (listImage[1].shape[0] >= listImage[1].shape[1]):
            listImage[1] = resizeImage(listImage[1], 400, int((listImage[1].shape[1]*400)/listImage[1].shape[0]))
        else:
            listImage[1] = resizeImage(listImage[1], int((listImage[1].shape[0]*400)/listImage[1].shape[1]), 400)

        if option == 1:
            listImage[1] = rgb2gray(listImage[1])
            actName = "Grayscale"
        elif option == 2:
            listImage[1] = rgb2bnw(listImage[1])
            actName = "Black & White"

        closeElement(3)
        if imgStats[1]: closeElement(2)
        imgStats[1] = True
        listImage[3] = ImageTk.PhotoImage(Image.fromarray(listImage[1]))
        listImage[4] = makeHistogram(listImage[1])

        if viewMode == 1:
            listLabel[0] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[2])
            listLabel[2] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[3])
        elif viewMode == 2:
            listLabel[0] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[3])
            listLabel[2] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[4])

        listLabel[0].pack(side="left", padx=10)
        listLabel[2].pack(side="right", padx=10)
        listLabel[1] = Label(tkFrame[0], padx=10, borderwidth=3, relief="ridge", width=101, text=actName, font="Helvetica 12 bold")        
        listLabel[1].pack(side="top")
        tkFrame[0].pack()
        tkFrame[1].pack(side="bottom", pady=10)
    else:
        messagebox.showwarning("Warning", "Open file first!")

def editAction(option):
    global imgPath
    global actName
    global listLabel
    global listImage
    global imgStats
    
    if imgStats[0]:
        listImage[1] = listImage[0]

        if (listImage[1].shape[0] >= listImage[1].shape[1]):
            listImage[1] = resizeImage(listImage[1], 400, int((listImage[1].shape[1]*400)/listImage[1].shape[0]))
        else:
            listImage[1] = resizeImage(listImage[1], int((listImage[1].shape[0]*400)/listImage[1].shape[1]), 400)
               
        if option == 11:
            ents = rgbInput(11)
            r = int(ents['Red'].get())
            g = int(ents['Green'].get())
            b = int(ents['Blue'].get())
            listImage[1] = addPixel(listImage[1], r, g, b)
            actName = "Addition"
        elif option == 12:
            ents = rgbInput(12)
            r = int(ents['Red'].get())
            g = int(ents['Green'].get())
            b = int(ents['Blue'].get())
            listImage[1] = minPixel(listImage[1], r, g, b)
            actName = "Substraction"
        elif option == 13:
            ents = rgbInput(13)
            r = int(ents['Red'].get())
            g = int(ents['Green'].get())
            b = int(ents['Blue'].get())
            listImage[1] = multPixel(listImage[1], r, g, b)
            actName = "Multiplication"
        elif option == 14:
            ents = rgbInput(14)
            r = int(ents['Red'].get())
            g = int(ents['Green'].get())
            b = int(ents['Blue'].get())
            listImage[1] = divPixel(listImage[1], r, g, b)
            actName = "Division"
        elif option == 2:
            listImage[1] = sampling(listImage[1], valInput(2))
            actName = "Sampling"
        elif option == 3:
            listImage[1] = quantization(listImage[1], valInput(3))
            actName = "Quantization"
        elif option == 41:
            listImage[1] = incrIntensity(listImage[1], valInput(41))
            actName = "Increase Intensity"
        elif option == 42:
            listImage[1] = decrIntensity(listImage[1], valInput(42))
            actName = "Decrease Intensity"
        elif option == 5:
            listImage[1] = inverse2(listImage[1])
            actName = "Inverse"
        elif option == 61:
            listImage[1] = histEqualization(listImage[1])
            actName = "Histogram Equalization"
        elif option == 62:
            ref_path = filedialog.askopenfilename(filetypes=[("JPEG File", "*.jpeg *.jpg"),("JPEG-2000", "*.jp2 .jpx"),("TIFF", "*.tiff"),("SVG", "*.svg"),("GIF", "*.gif"),("BMP", "*.bmp"), ("PNG", "*.png")])
            ref_img = np.asarray(Image.open(ref_path))
            listImage[1] = histSpecification(listImage[1], ref_img)
            actName = "Histogram Specification"
        elif option == 71:
            kernel = np.array([[1/16, 1/8, 1/16], 
                                [1/8, 1/4, 1/8],
                                [1/16, 1/8, 1/16]])
            listImage[1] = convFilter(kernel, listImage[1])
            actName = "Low Pass Filter"
        elif option == 72:
            kernel = np.array([[-1, -1, -1], 
                                [-1, 8, -1],
                                [-1, -1, -1]])
            listImage[1] = convFilter(kernel, listImage[1])
            actName = "High Pass Filter"
        elif option == 73:
            kernel = np.array([[-1, -1, -1], 
                                [-1, 9, -1],
                                [-1, -1, -1]])
            listImage[1] = convFilter(kernel, listImage[1])
            actName = "Band Pass Filter"
        elif option == 8:
            listImage[1] = listImage[0]
            actName = "Default"
            if (listImage[1].shape[0] >= listImage[1].shape[1]):
                listImage[1] = resizeImage(listImage[1], 400, int((listImage[1].shape[1]*400)/listImage[1].shape[0]))
            else:
                listImage[1] = resizeImage(listImage[1], int((listImage[1].shape[0]*400)/listImage[1].shape[1]), 400)
        
        closeElement(3)
        if imgStats[1]: closeElement(2)
        imgStats[1] = True
        listImage[3] = ImageTk.PhotoImage(Image.fromarray(listImage[1]))
        listImage[4] = makeHistogram(listImage[1])

        if viewMode == 1:
            listLabel[0] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[2])
            listLabel[2] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[3])
        elif viewMode == 2:
            listLabel[0] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[3])
            listLabel[2] = Label(tkFrame[1], borderwidth=3, relief="ridge", height=500, width=500, image=listImage[4])

        listLabel[0].pack(side="left", padx=10)
        listLabel[2].pack(side="right", padx=10)
        listLabel[1] = Label(tkFrame[0], padx=10, borderwidth=3, relief="ridge", width=101, text=actName, font="Helvetica 12 bold")        
        listLabel[1].pack(side="top")
        tkFrame[0].pack()
        tkFrame[1].pack(side="bottom", pady=10)
    else:
        messagebox.showwarning("Warning", "Open file first!")

def createMenubar(root):
    menubar = Menu(root)

    file = Menu(menubar, tearoff=0)  
    file.add_command(label="Open", command=openFile)  
    file.add_command(label="Save", command=saveFile)  
    file.add_command(label="Save as", command=saveAsFile)  
    file.add_command(label="Close", command=lambda: closeElement(1))    
    file.add_separator()  
    file.add_command(label="Exit", command=lambda: exitApp(root))
    menubar.add_cascade(label="File", menu=file)    

    convert = Menu(menubar, tearoff=0)
    convert.add_command(label="to Grayscale", command=lambda: convertAction(1))
    convert.add_command(label="to B&W", command=lambda: convertAction(2))
    menubar.add_cascade(label="Convert", menu=convert)

    arithmetic = Menu(menubar, tearoff=0)
    arithmetic.add_command(label="Addition", command=lambda:editAction(11))
    arithmetic.add_command(label="Substraction", command=lambda:editAction(12))
    arithmetic.add_command(label="Multiplication", command=lambda:editAction(13))
    arithmetic.add_command(label="Division", command=lambda:editAction(14))

    intensity = Menu(menubar, tearoff=0)
    intensity.add_command(label="Increase", command=lambda:editAction(41))
    intensity.add_command(label="Decrease", command=lambda:editAction(42))

    histogram = Menu(menubar, tearoff=0)
    histogram.add_command(label="Histogram Equalization", command=lambda:editAction(61))
    histogram.add_command(label="Histogram Specification", command=lambda:editAction(62))

    filter = Menu(menubar, tearoff=0)
    filter.add_command(label="Low Pass Filter", command=lambda:editAction(71))
    filter.add_command(label="High Pass Filter", command=lambda:editAction(72))
    filter.add_command(label="Band Pass Filter", command=lambda:editAction(73))

    mode = Menu(menubar, tearoff=0)
    mode.add_radiobutton(label="Mode-1", command=lambda:changeMod(1))
    mode.add_radiobutton(label="Mode-2", command=lambda:changeMod(2))

    edit = Menu(menubar, tearoff=0)
    edit.add_cascade(label="Pixel Operations", menu=arithmetic)
    edit.add_command(label="Sampling", command=lambda: editAction(2))  
    edit.add_command(label="Quantization", command=lambda:editAction(3)) 
    edit.add_cascade(label="Intensity Operations", menu=intensity) 
    edit.add_command(label="Inverse", command=lambda:editAction(5)) 
    edit.add_cascade(label="Histogram", menu=histogram)
    edit.add_cascade(label="Filter", menu=filter)
    edit.add_separator()  
    edit.add_command(label="Default", command=lambda: editAction(8)) 

    menubar.add_cascade(label="Edit", menu=edit)
    menubar.add_cascade(label="Mode", menu=mode)
    menubar.add_command(label="License", command=lambda: showLicense())

    return menubar 

# Global Variable #
imgPath = ""
actName = ""
viewMode = 1
tkFrame = ["",""]
listLabel = ["","",""]
listImage = ["","","","",""]
imgStats = [False, False]
fields = ('Red', 'Green', 'Blue')

# Main Program #
tkRoot = createRoot("ImagiNation", "icon/icon.ico")
menubar = createMenubar(tkRoot)
tkFrame[0] = Frame(tkRoot)
tkFrame[1] = Frame(tkFrame[0])
tkRoot.config(menu=menubar)  
tkRoot.mainloop() 
