def predict_rice(img_path):
    model = load_model("projectlt/rice_model.h5")
    class_names = ["Rice Type 1", "Rice Type 2", "Rice Type 3", "Rice Type 4", "Rice Type 5"]
    
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)]

test_image = "project1/Jasmine.jpg"  
if os.path.exists(test_image):
    result = predict_rice(test_image)
    print(f"Predicted Rice Type: {result}")
else:
    print("Test image not found!")
