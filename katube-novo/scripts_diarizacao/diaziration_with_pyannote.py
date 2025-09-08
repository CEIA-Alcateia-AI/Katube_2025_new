import os
import glob
import argparse
from tqdm import tqdm
import torch
import torchaudio
from pyannote.audio import Pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Running on {device}...")

def process_audio_files(input_dir, huggingface_token):
    """
    Process all WAV files in the specified directory and perform speaker diarization.

    Parameters:
    - input_dir (str): Path to the directory containing WAV files.
    - huggingface_token (str): Hugging Face access token for authentication.
    """
    # Check if the directory exists
    if not os.path.isdir(input_dir):
        print(f"The directory '{input_dir}' does not exist.")
        return

    # Instantiate the pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=huggingface_token
    ).to(device)

    # Get the list of WAV files in the directory
    wav_files = glob.glob(os.path.join(input_dir, "**/**/*.wav"))

    if not wav_files:
        print("No WAV files found in the directory.")
        return

    # Iterate over the files and process them
    for audio_path in tqdm(wav_files):
        file_name = os.path.basename(audio_path)
        #print(f"Processing '{file_name}'...")

        # Check if RTTM result file exists
        output_dir = os.path.dirname(audio_path)
        base_name = os.path.splitext(file_name)[0]
        rttm_path = os.path.join(output_dir, base_name + '.rttm')
        if os.path.exists(rttm_path):
            print(f"File {file_name} already exists.")
            continue

        try:
            # Run the pipeline on the audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.to(device)
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

            # Save the result to an RTTM file
            with open(rttm_path, "w") as rttm:
                diarization.write_rttm(rttm)
            #print(f"Result saved to '{rttm_path}'.")

        except Exception as e:
            print(f"An error occurred while processing '{file_name}': {e}")

def main():
    """
    Main function to parse arguments and initiate processing.
    """
    parser = argparse.ArgumentParser(description="Batch speaker diarization for WAV files in a directory.")
    parser.add_argument('directory', type=str, help='Path to the directory containing WAV files.')
    parser.add_argument('--huggingface_token', type=str, required=False, default="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", help='Hugging Face access token for authentication.')

    args = parser.parse_args()

    process_audio_files(args.directory, args.huggingface_token)

if __name__ == "__main__":
    main()

