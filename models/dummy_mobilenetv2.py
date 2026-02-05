from models.mobilenetv2 import build_mobilenetv2

# Task 1 → 448x448
model = build_mobilenetv2(input_shape=(448, 448, 3))

model.save("models/dummy/task1_spatial_dummy.keras")

print("✅ Dummy model saved")
