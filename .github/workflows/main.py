import sys, os
import cv2
import mediapipe as mp
import time
import tkinter as tk
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


#CHANGE TIMINGS AND PROBLEMS FUNCTION FOR DIFFERENT SENSITIVITY
# since it is the demo mode it is sensitive
stage = 0
def button_pressed():
   start.destroy()
   global stage
   stage += 1


if stage == 0:
   #change color! and texts
   start = tk.Tk()
   start.title("Welcome to PosturePerfect")
   start.geometry('550x300')
   messages = Label(start, text=f"Welcome to PosturePerfect!\nGet ready with good posture:\n- 50-70 cm away from the screen\n- Your head or shoulders are not tilted\n- Your ears and shoulders are in the same z plane. ", fg='#FF007F', font=('Helvetica', 24), bg='black')
   messages.grid(column=0, row=0, pady=50)


   btn = Button(start, text="Let's get started!", fg="black", font=('Helvetica', 20), command=button_pressed)
   btn.configure(bg="#4B0082")
   btn.grid(column=0, row=1)
   start.mainloop()


def newWin(issue):
  window = tk.Tk()
  window.title("Posture Moniter")
  window.geometry('450x300')


  #pass through error
  prob = []
  print(issue)
  if issue[0] == 0:
      prob.append('You are too close to the screen! ')
  if issue[1] == 0:
      prob.append('You are too slouched! ')


  elif issue[2] == 0:
      prob.append('Your sholders are tilted! ')


  elif issue[3] == 0:
      prob.append('Your face is tilted! ')


  elif issue[4] == 0:
      prob.append('You are leaning forward! ')


  # Add text label
  text_label = Label(window, text=f"Your posture has been getting worse!\n{' '.join(prob)}\nReady to go back and fix your posture?", fg='black', font=('Helvetica', 24), bg='#EF36E3')
  text_label.grid(column=0, row=0, pady=50)


  # Button
  btn = Button(window, text="Yes", fg="black", font=('Helvetica', 12), command=window.destroy)
  btn.configure(bg="#4B0082")
  btn.grid(column=0, row=1)


  window.mainloop()


# Load the saved model
model = load_model("model2.keras")


# Define a function to predict on a single image
def predict_image(frame, model):
  # Convert the frame to RGB (OpenCV uses BGR by default)
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # Resize the frame to the required input size of the model
  resized_frame = cv2.resize(rgb_frame, (150, 150))
  # Convert the resized frame to float32 and normalize
  img_array = resized_frame.astype('float32') / 255.0
  # Expand dimensions to create a batch of size 1
  img_array = np.expand_dims(img_array, axis=0)
  # Make prediction
  prediction = model.predict(img_array)
  return prediction[0][0]




def problems(idx, topThirtySix):
                
  close = 0
  badAvg = 0


  for q in range(1, 4):
      if topThirtySix[-q][idx] == 0:
          close += 1


  for h in topThirtySix:
      if h[idx] == 0:
          badAvg += 1


  #if the past 4 were bad or half or more of the ones in the past 3 mins were
  if close == 2 or badAvg >= 2:
      res = 0
  else:
      res = 1


  return(res)




mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
counter = 0
calcs = 0
landmark_indices = [0, 7, 8, 11, 12]  # Landmark indices to extract
moniter = False
calcPer = 0
status = []
remind = False
lastRemind = 0
issues = []


def avg(list):
  #calculate average of x, y and z's
  allxcords = []
  allycords = []
  allzcords = []

  avgList = []
   #numbered of ordered lists
  for i in range(0, len(list)):
      allxcords.append(list[i][0])
      allycords.append(list[i][1])
      allzcords.append(list[i][2])

  xavg = 0
  for i in allxcords:
      xavg += i
                
      xavg = round((xavg/len(allxcords)), 2)
      avgList.append(xavg)

  yavg = 0
  for i in allycords:
      yavg += i
                
      yavg = round((yavg/len(allycords)), 2)
      avgList.append(yavg)

  zavg = 0
  for i in allzcords:
      zavg += i
                
      zavg = round((zavg/len(allzcords)), 2)
      avgList.append(zavg)


  return avgList


if stage > 0:
   cap = cv2.VideoCapture(1)
   with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
       start = time.time()


       while cap.isOpened():
           ret, frame = cap.read()
           if not ret:
               break
          
           # Recolor image to RGB
           image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           image.flags.writeable = False
          
           # Make detection
           results = pose.process(image)
          
           # Recolor back to BGR
           image.flags.writeable = True
           image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
           background = np.zeros_like(frame)

           mp_drawing.draw_landmarks(background, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                       )


           # Prints certain landmarks and collects data
           if results.pose_landmarks:
               startlandmarks = []
               nose = []
               rface = []
               lface = []
               rshold = []
               lshold = []
               #for posture scale (0 is bad, 100 is perfecta!)
               current = time.time()




               # Only records starting at 5 seconds
               if (round(current)-round(start)) == 5 and counter == 0:




                   for idx in landmark_indices:
                       landmark = results.pose_landmarks.landmark[idx]
                       x, y, z = landmark.x, landmark.y, landmark.z
                       if idx == 0:
                           nose.append((x, y, z))
                       elif idx == 7:
                           rface.append((x, y, z))
                       elif idx == 8:
                           lface.append((x, y, z))
                       elif idx == 11:
                           rshold.append((x, y, z))
                       elif idx == 12:
                           lshold.append((x, y, z))

                       # Print the x, y, z coordinates of the landmark
                       print(f"Starting Landmark {idx}: x={x}, y={y}, z={z}")


                   #assume starting at good posture
                   status.append(1)


                       #lets it know its done
                   counter += 1

                       #So calculations are only done once
               if calcs == 0 and counter == 1:
                   #calculate average x, y, z values
                   avgNose = avg(nose)
                   avgrFace = avg(rface)
                   avglFace = avg(lface)
                   avgrShold = avg(rshold)
                   avglShold = avg(lshold)

                   #print averages!
                   print(avgNose, avgrFace, avglFace, avgrShold, avglShold)
                   calcs += 1
                   counter += 1


                   #average of starting shoulders
                   startSides = round(((avglFace[1]+ avgrFace[1])/2), 2)
                   startShold = round(((avglShold[1] + avgrShold[1])/2), 2)

                   startDis = round((startShold - startSides), 2)

               # compares landmarks every 5 seconds and adds to
               if moniter == False and (round(current)-round(start)) % 5 == 0 and counter > 1:


                   macPer = (predict_image(background, model))


                   #trying to create more variation in the lower values
                   if macPer < 0.09:
                       macPer = macPer*10


                   print(counter)
                   landmarkNose = results.pose_landmarks.landmark[0]
                   landmarkRFace = results.pose_landmarks.landmark[7]
                   landmarkLFace = results.pose_landmarks.landmark[8]
                   landmarkRShoulder = results.pose_landmarks.landmark[11]
                   landmarkLShoulder = results.pose_landmarks.landmark[12]



                   #average of current shoulders
                   curSides = round(((landmarkRFace.y + landmarkLFace.y)/2), 2)
                   curShold = round(((landmarkRShoulder.y + landmarkLShoulder.y)/2), 2)


                   #take distance between averagers
                   curDis = round((curShold - curSides), 2)


                   #check out zAllow
                   zAllow = 0.5
                   yAllow = 0.05
                   yshAllow = 0.2
                   zshAllow = 0.22


                   currentStat = []
                   #for calulation of current posture based on monitering all aspects (0 is bad, 1 is good)
                  
                   if round(landmarkNose.z, 2)+zAllow <=avgNose[2]:
                       #check if extra allowance is needed for slight movments!!! (for all measurements)
                       currentStat.append(0)
                       print("WOW! too close")
                   else:  
                       currentStat.append(1)


                   if curDis+yAllow < startDis:
                       currentStat.append(0)
                       print("WOW! too scrunched")
                   else:
                       currentStat.append(1)

                   #if left shoulder is outside of rights bounds
                   if not round(landmarkRShoulder.y, 2)-yshAllow < round(landmarkLShoulder.y, 2) < round(landmarkRShoulder.y, 2)+yshAllow:
                       print("titled shoulders")
                       currentStat.append(0)
                   else:
                       currentStat.append(1)

                   if not round(landmarkRFace.y, 2)-yshAllow < round(landmarkLFace.y, 2) < round(landmarkRFace.y, 2)+yshAllow:
                       print("face tilted")
                       currentStat.append(0)
                   else:
                       currentStat.append(1)


                   if not round(landmarkRShoulder.z, 2)+zshAllow >= round(landmarkLShoulder.z, 2) >= round(landmarkRShoulder.z, 2)-zshAllow:
                       print("shoulder leaning")
                       currentStat.append(0)
                   else:
                       currentStat.append(1)

                    #make one for moving head to look out right or left!
                   moniter = True
                   counter += 1


               #for calculating posture scale based on calculations
               #if they are too close or slouching then bad posture
                   print(currentStat)
                   #add it to the total issues list
                   issues.append(currentStat)

                   if currentStat[0] == 0:
                       calcPer = 0.1

                   elif currentStat[1] == 0:
                       calcPer = 0.1

                   elif currentStat[2] == 0 or currentStat[3] == 0 or currentStat[4] == 0:
                       calcPer = 0.75

                   elif (currentStat[2] == 0 and (currentStat[3] == 0 or currentStat[4] == 0)) or (currentStat[3] == 0 and (currentStat[2] == 0 or currentStat[4] == 0)) or (currentStat[4] == 0 and (currentStat[3] == 0 or currentStat[2] == 0)):
                       calcPer = 0.5
                  
                   # if 3/3 then 30% posture
                   elif currentStat[2] == 0 and currentStat[3] == 0 and currentStat[4] == 0:
                       calcPer = 0.3
                      
                   else:
                       calcPer = 1

                   print(f"calculation Per: {calcPer}")

                   #if posture is kinda bad
                   if macPer <= 0.5:

                       #calculated final posture score
                       postureScore = (0.4*calcPer)+(0.6*macPer)
                       status.append(round(postureScore, 2))


                   else:
                       #calculated final posture score
                       postureScore = (0.6*calcPer)+(0.4*macPer)
                       status.append(round(postureScore, 2))
                  
                   print(f"calc: {calcPer}, ml: {macPer}")
                   #status collects posture scores (0 is bad and 1 is good)
                   print(status)


                   #EDIT TIMING FOR DEMO
                   timings = 4


                   #only allows to check every 4 minutes
                   if counter >= timings and (lastRemind == 0 or counter-lastRemind >= timings):
                       scoreAvg = 0


                       for i in range (1, timings):
                           scoreAvg += status[-i]

                       scoreAvg = scoreAvg/(timings-1)


                       #if the posture is bad
                       if scoreAvg < 0.5:
                           remind = True
                           lastRemind = counter


                   if remind == True:
                       #edit so that currentStat is the average current stat for the past 36 runs
                       prob = []
                      
                       #changed to 2 for now!
                       topThirtySix = []
                       for i in range(1, timings):
                           topThirtySix.append(issues[-i])                         
                      
                       #calculate main issues and add them to prob
                       for u in range(0, 5):
                           res = problems(u, topThirtySix)
                           prob.append(res)

                       '''

                       #find average issues of each (if more than half of the past 10 runs had the same issue then add it to prob)
                       uMes = []
                       if prob[0] == 0:
                           uMes.append('You are too close to the screen! ')
                       if prob[1] == 0:
                           uMes.append('You are too slouched! ')
                       elif prob[2] == 0:
                           uMes.append('Your sholders are tilted! ')




                       elif prob[3] == 0:
                           uMes.append('Your face is tilted! ')




                       elif prob[4] == 0:
                           uMes.append('You are leaning forward! ')




                       print(f"\nYOU HAVE A PROBLEM! \nIT IS: {uMes}\n")


                       '''
                       newWin(prob)
                       remind = False
                      
               elif moniter and (round(current) - round(start)) % 5 > 0:
                   moniter = False


           # Display the image
           cv2.imshow('Mediapipe Feed', background)
              
           # Press 'q' to exit
           if cv2.waitKey(10) & 0xFF == ord('q'):
               break


       cap.release()
       cv2.destroyAllWindows()