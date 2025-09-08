import os
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
from glob import glob
import shutil

class AudioSegmentProcessor:
    def __init__(self, audio_path: str, dest_dir: str):
        self.audio_path = audio_path
        self.dest_dir = dest_dir
        self.audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        self.audio = AudioSegment.from_file(audio_path)
        self.min_duration = 2.5  # Minimum duration for a segment
        os.makedirs(self.dest_dir, exist_ok=True)

    @staticmethod
    def merge_segments(df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges consecutive segments belonging to the same speaker.

        Args:
            df (pd.DataFrame): DataFrame with columns ['START', 'END', 'SPEAKER', 'TEXT'].

        Returns:
            pd.DataFrame: Merged segments as a DataFrame.
        """
        merged_segments = []
        i = 0
        while i < len(df):
            start = df.iloc[i]['START']
            end = df.iloc[i]['END']
            speaker = df.iloc[i]['SPEAKER']
            text = df.iloc[i]['TEXT'] if 'TEXT' in df.columns else ""

            while (
                i + 1 < len(df) and
                df.iloc[i + 1]['SPEAKER'] == speaker and
                (df.iloc[i + 1]['START'] - end) <= 1.5 and  # Pause <= 1.5 seconds
                (df.iloc[i + 1]['END'] - start) <= 20  # Total duration <= 20 seconds
            ):
                end = df.iloc[i + 1]['END']
                if 'TEXT' in df.columns:
                    text += " " + df.iloc[i + 1]['TEXT']
                i += 1

            duration = end - start
            merged_segments.append([start, end, duration, speaker, text])
            i += 1

        return pd.DataFrame(merged_segments, columns=['START', 'END', 'DURATION', 'SPEAKER', 'TEXT'])


    def process_segments(self, df: pd.DataFrame) -> str:
        """
        Processes the merged segments, saves valid audio segments, and generates a CSV file.

        Args:
            df (pd.DataFrame): DataFrame with columns ['START', 'END', 'SPEAKER', 'TEXT'].

        Returns:
            str: Path to the generated CSV file.
        """
        # Merge segments first
        df_merged = self.merge_segments(df)
        print(f"Merged {len(df)} segments into {len(df_merged)} segments.")

        if len(df_merged) == 1:
            print("No speaker segments to process.")
            #shutil.copy(self.audio_path, os.path.join(self.dest_dir, os.path.basename(self.audio_path)))
            return
        
        # List to store segment information
        segment_data = []

        result = True
        for row in tqdm(df_merged.itertuples(index=False), total=len(df_merged), desc="Processing segments"):
            # Convert times to milliseconds
            start_time = int(row.START * 1000)
            end_time = int(row.END * 1000)
            duration = end_time - start_time

            # Skip segments that are too short
            if duration < self.min_duration * 1000:
                continue

            # Validate segment bounds
            if start_time >= len(self.audio) or end_time > len(self.audio):
                print(f"Skipping segment: {start_time}-{end_time} (out of bounds)")
                continue

            # Extract the audio segment
            audio_segment = self.audio[start_time:end_time]

            # Create directory for the speaker
            speaker = row.SPEAKER

            # Generate filename and export the audio segment
            segment_filename = f"{self.audio_name}_{speaker}_{start_time}_{end_time}.wav"
            #
            segment_path = os.path.join(self.dest_dir, segment_filename)

            try:
                audio_segment.export(segment_path, format="wav")
            except Exception as e:
                print(f"Error exporting segment: {segment_filename}")
                print(e)
                result = False
                continue

            '''
            # Add segment information to the list
            text = row.TEXT if hasattr(row, 'TEXT') else ""
            segment_data.append({
                "path": segment_path,
                "speaker": speaker,
                "text": text,
                "duration": duration
            })
            '''            
        if result:
            print(f"Segments saved at: {self.dest_dir}")
            os.remove(self.audio_path)



def rttm_to_dataframe(rttm_file: str) -> pd.DataFrame:
    """
    Converts an RTTM file into a pandas DataFrame.

    Args:
        rttm_file (str): Path to the RTTM file.

    Returns:
        pd.DataFrame: DataFrame containing the RTTM information with relevant columns.
    """
    # Define the column names as per RTTM format
    columns = [
        "TYPE", "FILENAME", "CHANNEL", "START", "DURATION",
        "ORTHO", "SUBTYPE", "SPEAKER", "CONFIDENCE", "MISC"
    ]

    # Initialize an empty list to store rows
    rows = []

    # Read the RTTM file and parse each line
    with open(rttm_file, 'r') as file:
        for line in file:
            fields = line.strip().split()
            if len(fields) == len(columns):
                rows.append(fields)
            else:
                print(f"Skipping line with incorrect format: {line.strip()}")

    if not rows:
        print(f"No valid data found in {rttm_file}")
        return pd.DataFrame(columns=columns)

    # Convert the rows into a DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Convert relevant columns to appropriate data types
    df["START"] = pd.to_numeric(df["START"], errors="coerce")
    df["DURATION"] = pd.to_numeric(df["DURATION"], errors="coerce")
    df["CHANNEL"] = pd.to_numeric(df["CHANNEL"], errors="coerce")

    # Calculate END time
    df['END'] = df['START'] + df['DURATION']

    return df


def main(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    rttm_files = glob(os.path.join(input_dir, "*.rttm"))
    for filepath in tqdm(rttm_files, desc="Processing RTTM files"):
        audio_filepath = os.path.splitext(filepath)[0] + ".wav"
        if not os.path.exists(audio_filepath):
            print(f"Audio file not found for {filepath}, skipping.")
            continue

        diarized_dataframe = rttm_to_dataframe(filepath)
        if diarized_dataframe.empty:
            print(f"No data to process for {filepath}, skipping.")
            continue

        processor = AudioSegmentProcessor(audio_filepath, output_dir)
        processor.process_segments(diarized_dataframe)


if __name__ == "__main__":
    # Replace these paths with your actual directories
    input = "podcasts_segmented_enhanced/0"
    for folder in  tqdm(glob(os.path.join(input, "**/**"))):
         main(folder, folder)
