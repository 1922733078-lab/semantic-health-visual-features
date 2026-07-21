"""Extract features for banner/poster/packaging using same algo as fast_features.py"""
import numpy as np, pandas as pd, cv2
from pathlib import Path
from scipy.stats import entropy
from tqdm import tqdm

def extract(p):
    try:
        img = cv2.imread(str(p))
        if img is None: return None
        img = cv2.resize(img, (256, 256))
    except:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    f = {}
    # Color
    pixels = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1,3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(pixels, 5, None, crit, 5, cv2.KMEANS_RANDOM_CENTERS)
    sizes = np.bincount(labels.flatten(), minlength=5)
    f["num_dominant_colors"] = int(np.sum(sizes/len(labels) > 0.05))
    h,s,v = hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    f["hue_mean"]=float(h.mean()); f["saturation_mean"]=float(s.mean())
    f["saturation_std"]=float(s.std()); f["value_mean"]=float(v.mean())
    f["value_std"]=float(v.std())
    L=lab[:,:,0].astype(np.float64)
    f["lightness_contrast"]=float(np.sqrt(np.mean((L-L.mean())**2)))
    hh=np.histogram(h.flatten(),bins=36,range=(0,180))[0].astype(float)
    hh=hh/(hh.sum()+1e-10)
    f["color_entropy"]=float(entropy(hh+1e-10))
    f["warm_color_ratio"]=float(((h<=30)|(h>=150)).mean())
    q=np.argsort(hh)[-2:]
    hd=min(abs(int(q[0])-int(q[1]))*5,180-abs(int(q[0])-int(q[1]))*5)
    f["color_harmony"]=1.0-min(abs(hd),abs(hd-30),abs(hd-60),abs(hd-120),abs(hd-180))/90.0
    th=np.argsort(hh)[-3:]; hds=[]
    for i in range(len(th)):
        for j in range(i+1,len(th)):
            d=abs(int(th[i])-int(th[j]))*5
            hds.append(min(d,180-d)/90.0)
    f["hue_contrast"]=float(np.mean(hds)) if hds else 0.0
    # Texture
    edges=cv2.Canny(gray,50,150)
    f["edge_density"]=float(edges.mean()/255.0)
    sx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    sy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    mag=np.sqrt(sx**2+sy**2)
    ori=np.arctan2(sy,sx)
    strong=mag>np.percentile(mag,70)
    if strong.sum()>0:
        oh=np.histogram(ori[strong],bins=8,range=(-np.pi,np.pi))[0].astype(float)
        oh=oh/(oh.sum()+1e-10)
        f["edge_orientation_entropy"]=float(entropy(oh+1e-10))
    else:
        f["edge_orientation_entropy"]=0.0
    gh=cv2.calcHist([gray],[0],None,[64],[0,256])
    gh=gh/(gh.sum()+1e-10)
    f["gray_mean"]=float(gray.mean())
    f["gray_std"]=float(gray.std())
    f["gray_entropy"]=float(entropy(gh.flatten()+1e-10))
    f["gradient_energy"]=float(mag.mean())
    # Composition
    flip=cv2.flip(gray,1)
    f["symmetry"]=float(1.0-np.mean(np.abs(gray.astype(float)-flip.astype(float))/255.0))
    h_g,w_g=gray.shape
    total=gray.sum()
    if total>0:
        cy=np.sum(gray*np.arange(h_g)[:,np.newaxis])/total
        cx=np.sum(gray*np.arange(w_g)[np.newaxis,:])/total
    else:
        cy,cx=h_g/2,w_g/2
    th_,tw=h_g/3,w_g/3
    ds=[np.sqrt((cx-tw)**2+(cy-th_)**2),np.sqrt((cx-2*tw)**2+(cy-th_)**2),
        np.sqrt((cx-tw)**2+(cy-2*th_)**2),np.sqrt((cx-2*tw)**2+(cy-2*th_)**2)]
    f["rule_of_thirds"]=float(1.0-min(ds)/np.sqrt(h_g**2+w_g**2))
    f["center_offset_x"]=float((cx-w_g/2)/(w_g/2))
    f["center_offset_y"]=float((cy-h_g/2)/(h_g/2))
    es=cv2.resize(edges.astype(np.float32),(32,32))
    f["whitespace_ratio"]=float((es<10).mean())
    f["fg_bg_ratio"]=float((gray>gray.mean()).mean())
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    mag2=np.sqrt(gx**2+gy**2)
    hh2,ww2=gray.shape[:2]
    diag=np.eye(hh2,ww2,dtype=bool)
    anti=np.fliplr(np.eye(hh2,ww2,dtype=bool))
    dm=mag2[diag].mean()
    am=mag2[anti].mean()
    f["diagonal_energy_ratio"]=float(dm/(am+1e-10)) if (dm+am)>0 else 1.0
    # Typography
    f["text_coverage"]=float(edges.mean()/255.0)
    hp=gray.mean(axis=1)
    f["text_block_count"]=int(np.sum(np.diff(hp>hp.mean())>0))
    bs=[]
    for y in range(0,256-32,32):
        for x in range(0,256-32,32):
            bs.append(float(gray[y:y+32,x:x+32].std()))
    f["font_size_cv"]=float(np.std(bs)/np.mean(bs)) if bs and np.mean(bs)>0 else 0.0
    _,bn=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    f["has_text"]=1 if 0.02<(bn==0).mean()<0.5 else 0
    # Saliency (spectral residual)
    ft=np.fft.fft2(gray.astype(np.float32))
    sh=np.fft.fftshift(ft)
    mg=np.abs(sh)
    ph=np.angle(sh)
    lm=np.log(mg+1e-10)
    avg=cv2.boxFilter(lm,-1,(5,5))
    res=lm-avg
    sal=np.abs(np.fft.ifft2(np.fft.ifftshift(np.exp(res+1j*ph))))
    sal=(sal-sal.min())/(sal.max()-sal.min()+1e-10)
    f["saliency_mean"]=float(sal.mean())
    f["saliency_std"]=float(sal.std())
    return f


if __name__ == "__main__":
    all_new = []
    for cat in ["banner", "poster", "packaging"]:
        imgs = sorted(Path(f"data/processed/{cat}").glob("*.jpg"))
        print(f"Processing {cat}: {len(imgs)} images")
        for i, p in enumerate(tqdm(imgs, desc=cat)):
            ft = extract(p)
            if ft:
                ft["image_id"] = f"{cat}_{i:04d}"
                ft["category"] = cat
                all_new.append(ft)

    new_df = pd.DataFrame(all_new)
    old_df = pd.read_csv("data/features/traditional_features.csv")
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined.to_csv("data/features/traditional_features.csv", index=False)
    print(f"\nDone: {len(old_df)} existing + {len(new_df)} new = {len(combined)} total")
    print(f"  banner: {len(new_df[new_df.category=='banner'])}")
    print(f"  poster: {len(new_df[new_df.category=='poster'])}")
    print(f"  packaging: {len(new_df[new_df.category=='packaging'])}")
