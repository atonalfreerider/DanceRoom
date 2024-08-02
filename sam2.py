from ultralytics import SAM2


class Sam2:
    def __init__(self):
        self.model = SAM2('sam2_l.pt')

    def predict(self, video_path, output_dir):
        # only track people
        for i, pred in enumerate(self.model.track(source=video_path, labels=['person'], stream=True, persist=False)):
            pred.save(f"{output_dir}/frame_{i:04d}.png")
            print(f"Saved frame {i:04d} to {output_dir}/frame_{i:04d}.png")
