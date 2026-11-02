from app.ai.preprocessing import preprocess_image


image_path = "uploads/New_Doc_06-24-2026_17.17_page-0001.jpg"

image_array = preprocess_image(image_path)

print("Preprocessing successful!")
print("Shape:", image_array.shape)
print("Data type:", image_array.dtype)
print("Minimum value:", image_array.min())
print("Maximum value:", image_array.max())