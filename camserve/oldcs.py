#!/usr/bin/env python3
import os,sys,time,uuid,json,logging,datetime,threading
from typing import Optional
import cv2
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO

# Optional imports
CWD=os.path.dirname(os.path.realpath(__file__))
if os.path.join(CWD,"services") not in sys.path: sys.path.insert(0,os.path.join(CWD,"services"))
try: from detection_service import DetectionService
except: DetectionService=None
try: from deepface import DeepFace
except: DeepFace=None

logging.basicConfig(level=logging.INFO,format="[%(levelname)s] %(message)s")
log=logging.getLogger("camserv")

# Config
class C:
    EVENTS_FILE="events.json"
    CAPTURES_DIR="captured_images"
    RECORDINGS_DIR="captured_videos"
    FACES_DIR="faces"
    CAM_WIDTH,CAM_HEIGHT=640,480
    INFERENCE_SIZE=320
    PROCESS_INTERVAL=5
    MOTION_TRIGGER_SCAN_DURATION=10
    MOTION_THRESHOLD=30
    MOTION_MIN_AREA=500
    OBJECT_DETECTION_MODEL="yolov8n.pt"
    POSE_ESTIMATION_MODEL="yolov8n-pose.pt"
    FIRE_SMOKE_MODEL_PATH="packages/inferno_ncnn_model"
    SURVEILLANCE_CLASSES=["person","car"]
    EVENT_COOLDOWN=15

def ensure_dirs(*dirs):
    for d in dirs: os.makedirs(d,exist_ok=True)

# Video Recorder
class VR:
    def __init__(self,fps=20,duration=10,saved_folder=C.RECORDINGS_DIR):
        self.fps,self.duration,self.saved_folder=fps,duration,saved_folder
        ensure_dirs(saved_folder)
        self._lock=threading.Lock();self._writer=None;self._is_recording=False
        self._start_time=0;self._filepath=None
    def start(self,frame)->Optional[str]:
        with self._lock:
            if self._is_recording or frame is None: return None
            h,w=frame.shape[:2];ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._filepath=f"event_video_{ts}.mp4";self._writer=cv2.VideoWriter(os.path.join(self.saved_folder,self._filepath),cv2.VideoWriter_fourcc(*"mp4v"),self.fps,(w,h))
            if not self._writer.isOpened(): self._writer=None;return None
            self._is_recording=True;self._start_time=time.time()
            threading.Thread(target=self._timeout,daemon=True).start()
            log.info("Started recording %s",self._filepath)
            return self._filepath
    def _timeout(self):
        while self._is_recording and time.time()-self._start_time<self.duration: time.sleep(0.2)
        self.stop();log.info("Recording stopped (%s)",self._filepath)
    def add(self,frame):
        with self._lock:
            if self._is_recording and self._writer is not None:
                try: self._writer.write(frame)
                except: pass
    def stop(self):
        with self._lock:
            if self._writer: self._writer.release();self._writer=None
            self._is_recording=False;self._filepath=None

# Face/Emotion
class PR:
    def __init__(self,faces_dir=C.FACES_DIR):
        self.faces_dir=faces_dir;self.known=self._load()
    def _load(self): 
        if not os.path.isdir(self.faces_dir): return []
        return [(os.path.splitext(f)[0],os.path.join(self.faces_dir,f)) for f in os.listdir(self.faces_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    def recognize(self,frame,box):
        if DeepFace is None or not self.known: return "Unknown"
        x1,y1,x2,y2=box;roi=frame[y1:y2,x1:x2]
        if roi.size==0: return "Unknown"
        try:
            for name,path in self.known:
                r=DeepFace.verify(roi,path,enforce_detection=False,model_name="VGG-Face",distance_metric="cosine")
                if r.get("verified"): return name
        except: pass
        return "Unknown"

class ER:
    def analyze(self,frame,box):
        if DeepFace is None: return "N/A"
        x1,y1,x2,y2=box;roi=frame[y1:y2,x1:x2]
        if roi.size==0: return "N/A"
        try: res=DeepFace.analyze(roi,actions=["emotion"],enforce_detection=False);res=res[0] if isinstance(res,list) else res; return res.get("dominant_emotion","N/A")
        except: return "N/A"

# Main Server
class S:
    def __init__(self,cid=0):
        self.cid=cid;self.cap=None;self.running=False
        self._frame_lock=threading.Lock();self.frame=None;self.raw=None
        self.mods={"surv":False,"mon":False};self._last_event=0
        self.vr=VR();self.ds=None;self.ds_loaded=False
        self.od=None;self.pose=None;self.pr=None;self.er=None
        self.motion_end=0;self.bs=cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=50,detectShadows=False)
        self.fall_detected_time = None  # Tracks the start time of a fall
        self.fall_alert_triggered = False # Ensures we only send one alert per fall
        self.fc=0;self._file_lock=threading.Lock()
        ensure_dirs(C.CAPTURES_DIR,C.RECORDINGS_DIR,C.FACES_DIR)
        self.app=Flask(__name__);self._routes()
    def _routes(self):
        @self.app.route("/stream")
        def sgen():
            def gen():
                while self.running:
                    with self._frame_lock:
                        if self.frame is None: continue
                        ok,buf=cv2.imencode(".jpg",self.frame,[cv2.IMWRITE_JPEG_QUALITY,85])
                        if not ok: continue
                        data=buf.tobytes()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+data+b"\r\n")
                    time.sleep(0.033)
            return Response(gen(),mimetype="multipart/x-mixed-replace; boundary=frame")
        @self.app.route("/api/status")
        def status(): return jsonify(self.info())
        @self.app.route("/api/detection/config",methods=["POST"])
        def conf():
            cfg=request.get_json(silent=True) or {}
            self.mods["surv"]=bool(cfg.get("surveillance",self.mods["surv"]))
            self.mods["mon"]=bool(cfg.get("monitor",self.mods["mon"]))
            return jsonify({"success":True,"active_modules":self.mods})
    # Camera
    def start(self):
        if self.running: return True
        self.cap=cv2.VideoCapture(self.cid,cv2.CAP_DSHOW)
        if not self.cap.isOpened(): self.cap=cv2.VideoCapture(self.cid)
        if not self.cap.isOpened(): log.error("Camera fail"); return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,C.CAM_WIDTH);self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,C.CAM_HEIGHT)
        self.running=True;threading.Thread(target=self._loop,daemon=True).start();log.info("Camera started"); return True
    def stop(self):
        self.running=False;self.vr.stop();time.sleep(0.2)
        if self.cap: self.cap.release();self.cap=None;log.info("Camera stopped")
    # Models
    def load_ds(self):
        if self.ds_loaded: return self.ds is not None
        self.ds_loaded=True
        if DetectionService is None or not os.path.exists(C.FIRE_SMOKE_MODEL_PATH): return False
        try: self.ds=DetectionService(C.FIRE_SMOKE_MODEL_PATH);return True
        except: self.ds=None; return False
    def _ensure_od(self):
        if self.od is None:
            try: self.od=YOLO(C.OBJECT_DETECTION_MODEL)
            except: self.od=None
    def _ensure_pose(self):
        if self.pose is None:
            try: self.pose=YOLO(C.POSE_ESTIMATION_MODEL);self.pr=PR();self.er=ER()
            except: self.pose=None
    # Event
    def _save_event(self,data:dict):
        ts=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fn=f"event_{ts}.jpg";fp=os.path.join(C.CAPTURES_DIR,fn)
        with self._file_lock:
            try: cv2.imwrite(fp,self.raw);data["image_path"]=fn
            except: pass
            try:
                events_data = {"events": []}
                if os.path.exists(C.EVENTS_FILE) and os.path.getsize(C.EVENTS_FILE) > 0:
                    with open(C.EVENTS_FILE, "r") as f:
                        # Handle case where file might be empty or not a valid JSON
                        try:
                            content = json.load(f)
                            if isinstance(content, dict) and "events" in content:
                                events_data = content
                        except json.JSONDecodeError:
                            pass # Keep events_data as default empty
                
                events_data["events"].insert(0, data)
                
                with open(C.EVENTS_FILE,"w") as f: json.dump(events_data,f,indent=2)
            except Exception as e: log.error(f"Failed to save event: {e}")
    # Loop
    def _loop(self):
        while self.running and self.cap:
            ret,frame=self.cap.read(); 
            if not ret or frame is None: time.sleep(0.05); continue
            self.fc+=1;self.raw=frame.copy();proc=frame.copy();self.vr.add(self.raw)
            on_cd=(time.time()-self._last_event)<C.EVENT_COOLDOWN
            if self.mods["mon"]: proc=self._monitor(proc,on_cd)
            if self.mods["surv"]: proc=self._surv(proc,on_cd)
            with self._frame_lock: self.frame=proc
    # Surveillance
    def _surv(self,frame,on_cd):
        fg=self.bs.apply(frame);_,fg=cv2.threshold(fg,C.MOTION_THRESHOLD,255,cv2.THRESH_BINARY)
        fg=cv2.dilate(fg,None,iterations=2);cts,_=cv2.findContours(fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        motion=False;areas=[]
        for c in cts:
            if cv2.contourArea(c)<C.MOTION_MIN_AREA: continue
            motion=True;areas.append(cv2.contourArea(c))
            x,y,w,h=cv2.boundingRect(c);cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        scan=time.time()<self.motion_end
        if motion and not scan and not on_cd:
            self.motion_end=time.time()+C.MOTION_TRIGGER_SCAN_DURATION;self._last_event=time.time()
            vid=self.vr.start(self.raw)
            self._save_event({"id":str(uuid.uuid4()),"timestamp":datetime.datetime.now().isoformat(),"module":"surv","event_type":"Motion","video_path":vid,"motion_areas":areas, "notification": True})
            scan=True
        if scan:
            self.load_ds();self._ensure_od()
            if self.ds:
                try:
                    f,s=self.ds.detect_fire(frame)[1],self.ds.detect_smoke(frame)[1]
                    if f and not on_cd: self._last_event=time.time(); self._save_event({"id":str(uuid.uuid4()),"timestamp":datetime.datetime.now().isoformat(),"module":"surv","event_type":"Fire", "notification": True})
                    if s and not on_cd: self._last_event=time.time(); self._save_event({"id":str(uuid.uuid4()),"timestamp":datetime.datetime.now().isoformat(),"module":"surv","event_type":"Smoke"})
                except: pass
            if self.od:
                try:
                    r=self.od(cv2.resize(frame,(C.INFERENCE_SIZE,C.INFERENCE_SIZE)),verbose=False)
                    for res in r:
                        if not getattr(res,"boxes",None): continue
                        for b in res.boxes:
                            cidx=int(b.cls[0]);cls=res.names.get(cidx,str(cidx))
                            if cls in C.SURVEILLANCE_CLASSES:
                                xy=b.xyxy[0].cpu().numpy()
                                x1=int(xy[0]*C.CAM_WIDTH/C.INFERENCE_SIZE);y1=int(xy[1]*C.CAM_HEIGHT/C.INFERENCE_SIZE)
                                x2=int(xy[2]*C.CAM_WIDTH/C.INFERENCE_SIZE);y2=int(xy[3]*C.CAM_HEIGHT/C.INFERENCE_SIZE)
                                conf=float(b.conf[0].cpu().item());cv2.rectangle(frame,(x1,y1),(x2,y2),(0,165,255),2)
                except: pass
            rem=max(0,self.motion_end-time.time()); cv2.putText(frame,f"AI Scanning: {rem:.1f}s",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        return frame
    # Monitor
    def _monitor(self,frame,on_cd):
        self._ensure_pose(); 
        if self.pose is None: return frame
        
        try: res=self.pose(cv2.resize(frame,(C.INFERENCE_SIZE,C.INFERENCE_SIZE)),verbose=False)
        except: return frame
        
        is_currently_falling=False;boxes=[]
        
        for r in res:
            if not getattr(r,"boxes",None): continue
            for b in r.boxes:
                xywh=b.xywh[0].cpu().numpy();w,h=float(xywh[2]),float(xywh[3])
                xy=b.xyxy[0].cpu().numpy()
                x1=int(xy[0]*C.CAM_WIDTH/C.INFERENCE_SIZE);y1=int(xy[1]*C.CAM_HEIGHT/C.INFERENCE_SIZE)
                x2=int(xy[2]*C.CAM_WIDTH/C.INFERENCE_SIZE);y2=int(xy[3]*C.CAM_HEIGHT/C.INFERENCE_SIZE)
                
                # Check aspect ratio for fall detection
                is_fallen_pose = w > h * 1.4
                color=(0,0,255) if is_fallen_pose else (0,255,0)
                activity="Falling" if is_fallen_pose else "Stable"
                
                boxes.append([x1,y1,x2,y2]);cv2.rectangle(frame,(x1,y1),(x2,y2),color,2);cv2.putText(frame,activity,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
                
                if is_fallen_pose: is_currently_falling=True

        # --- NEW 10-SECOND TIMER LOGIC ---
        if is_currently_falling:
            # If this is the first frame we see a fall, start the timer.
            if self.fall_detected_time is None:
                self.fall_detected_time = time.time()
                log.info("Potential fall detected. Starting 10-second timer...")
            
            elapsed = time.time() - self.fall_detected_time
            
            # If fall persists for 10+ seconds and we haven't sent an alert yet...
            if elapsed >= 10 and not self.fall_alert_triggered and not on_cd:
                log.warning("EMERGENCY: Fall confirmed for over 10 seconds!")
                self._last_event=time.time()
                self.fall_alert_triggered = True # Mark alert as sent
                vid=self.vr.start(self.raw)
                
                # Save event WITH notification flag for the messaging system
                self._save_event({
                    "id":str(uuid.uuid4()),
                    "timestamp":datetime.datetime.now().isoformat(),
                    "module":"mon",
                    "event_type":"Fall", # Changed to event_type to match messaging script
                    "notification": True, # This is the key to trigger the SMS
                    "video_path":vid
                })
        else:
            # If no one is falling, reset the timer and alert flag.
            if self.fall_detected_time is not None:
                log.info("Fall condition ended. Resetting timer.")
            self.fall_detected_time = None
            self.fall_alert_triggered = False

        # --- END OF NEW LOGIC ---

        if boxes and self.fc%C.PROCESS_INTERVAL==0:
            for b in boxes:
                name=self.pr.recognize(frame,b) if self.pr else "Unknown"
                emo=self.er.analyze(frame,b) if self.er else "N/A"
                cv2.putText(frame,f"{name} ({emo})",(b[0],b[3]+20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        
        return frame
    def info(self): return {"running":self.running,"mods":self.mods,"ds_loaded":self.ds_loaded,"ds_ok":self.ds is not None,"models":{"od":self.od is not None,"pose":self.pose is not None}}
    def run_server(self,host="0.0.0.0",port=5001,debug=False):
        log.info("Server on %s:%s",host,port);self.app.run(host=host,port=port,debug=debug,threaded=True)

if __name__=="__main__":
    log.info("camserv start");log.info("fire path %s",C.FIRE_SMOKE_MODEL_PATH)
    srv=S(0);srv.mods["surv"]=True
    if srv.start():
        try: srv.run_server()
        except KeyboardInterrupt: log.info("kbd"); srv.stop()
    else: log.error("start fail")