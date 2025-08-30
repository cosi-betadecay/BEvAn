import ROOT as M
import math

# Load MEGAlib into ROOT
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

# Initialize MEGAlib
G = M.MGlobal()
G.Initialize()

# We are good to go ...


GeometryName = "$(MEGALIB)/resource/examples/geomega/special/Max.geo.setup"
FileName = "Activation.sim"

# Load geometry:
Geometry = M.MDGeometryQuest()
if Geometry.ScanSetupFile(M.MString(GeometryName)) == True:
  print("Geometry " + GeometryName + " loaded!")
else:
  print("Unable to load geometry " + GeometryName + " - Aborting!")
  quit()
    

Reader = M.MFileEventsSim(Geometry)
if Reader.Open(M.MString(FileName)) == False:
  print("Unable to open file " + FileName + ". Aborting!")
  quit()

NumberGoodEvents = 0
NumberBackgroundEvents = 0
while True: 
  Event = Reader.GetNextEvent()
  if not Event:
    break
  M.SetOwnership(Event, True)
  
  # Step 1: Find annihilation events
  NumberANNI = 0
  for i in range(0, Event.GetNIAs()):
    if Event.GetIAAt(i).GetProcess() == M.MString("ANNI"):
      NumberANNI += 1
      ProcessID = i+1
      # Find all IDs generated from this ANNI
      SecondaryIDs = []
      for i in range(0, Event.GetNIAs()):
        if Event.GetIAAt(i).GetOriginID() == ProcessID:
          SecondaryIDs.append(i+1)
      # Now check all HTs - if these add up to ~511
      TotalEnergy = 0
      for h in range(0, Event.GetNHTs()):
        for SID in SecondaryIDs:
          if Event.GetHTAt(h).IsOrigin(SID):
            TotalEnergy += Event.GetHTAt(h).GetEnergy()
            break;
      if math.fabs(TotalEnergy - 511) < 5:
        #print(f"Good annihilation event {TotalEnergy}")
        NumberGoodEvents += 1

  if NumberANNI == 0:
    #print("Not an annihilation event")
    NumberBackgroundEvents += 1

print(f"Number good 511 events: {NumberGoodEvents}")
print(f"Number background events: {NumberBackgroundEvents}")