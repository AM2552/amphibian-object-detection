from ultralytics import YOLO

images = [
    "datasets/amphibia/test/alpensalamander/0a21323f-72e3-4e2d-bfea-7546b1dc30ad.jpeg",
    "datasets/amphibia/test/bergmolch/0df04ace-fcd1-43ae-895b-ba1c44e35199.jpeg",
    "datasets/amphibia/test/erdkröte/1ab2f8aa-c99a-4df1-ab81-203685c24a84.jpg",
    "datasets/amphibia/test/feuersalamander/0b938c64-8211-47e3-860d-a65ffa0454db.jpeg",
    "datasets/amphibia/test/gelbbauchunke/0bf2068c-c2b3-4632-b1b6-0771311e437d.jpeg",
    "datasets/amphibia/test/grasfrosch/1ca70c3c-7fbb-457c-ace2-fc606928caa6.jpeg",
    "datasets/amphibia/test/kammmolch/0a25b11e-c116-40a9-8ea8-83636daf7c52.jpeg",
    "datasets/amphibia/test/knoblauchkröte/0e14bec7-8c5e-46d4-ae95-764e14b81b52.jpg",
    "datasets/amphibia/test/kreuzkröte/0b1465d8-762c-442d-931c-5bb9a7225108.jpeg",
    "datasets/amphibia/test/laubfrosch/1e07e594-7814-4b9b-9e09-c8e7d61aea84.jpeg",
    "datasets/amphibia/test/moorfrosch/0cd9a77f-9c92-4666-9dab-93f18732af9e.jpeg",
    "datasets/amphibia/test/rotbauchunke/0c6eaedf-3431-47cc-82f2-868481db0782.jpeg",
    "datasets/amphibia/test/springfrosch/0bfc7a2a-f1fa-48e5-86f2-3b7527b45af1.jpeg",
    "datasets/amphibia/test/teichmolch/0c90b2bb-6f14-42da-8cb3-7c45509fcb24.jpg",
    "datasets/amphibia/test/wasserfrosch/1ad38f51-56ee-4807-8654-83363eda2e68.jpeg",
    "datasets/amphibia/test/wechselkröte/2e5f978f-62f9-48bf-81dc-79e62b75de29.jpeg"
    
]

model = YOLO('amphibians/yolo_training/best.pt')
results = model(images)

for img in images:
    results = model(img)
    print(results)