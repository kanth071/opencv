import cv2
# Change IP if needed
# rtsp_url = "rtsp://admin:Cpplus-123@[192.168.3.250]:554/cam/realmonitor?channel=1&subtype=0"
rtsp_url = "rtsp://admin:Cpplus-123@192.168.3.250:554/cam/realmonitor?channel=2&subtype=0"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit()

print("✅ RTSP connected")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ No frame received")
        break

    cv2.imshow("CP Plus Camera", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()