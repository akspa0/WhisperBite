import os
import yaml
import logging

def write_outputs(audio_obj, config=None):
    """
    Write master transcript (TXT) and results summary (YAML) for the processed audio object.
    Expects audio_obj.metadata to contain 'input_file', 'preset', 'processing_time', and 'segments' (list of dicts).
    """
    logger = logging.getLogger(__name__)
    output_dir = config.get('output_dir', '.')
    results = audio_obj.metadata.copy()

    # --- Write Master Transcript ---
    master_transcript_path = os.path.join(output_dir, "master_transcript.txt")
    try:
        with open(master_transcript_path, "w", encoding='utf-8') as f:
            f.write(f"Master Transcript for: {os.path.basename(results.get('input_file', ''))}\n")
            f.write(f"Preset: {results.get('preset', '')}\n")
            f.write(f"Processed: {results.get('processing_time', '')}\n\n")
            segments = results.get('segments', [])
            if segments:
                segments_to_write = sorted(segments, key=lambda x: x.get('start_original', 0))
                f.write(f"Segments Processed: {len(segments_to_write)}\n\n")
                f.write("="*20 + "\n\n")
                for seg_res in segments_to_write:
                    segment_index = seg_res.get("segment_index", "N/A")
                    start_time = seg_res.get("start_original", 0)
                    end_time = seg_res.get("end_original", 0)
                    status = seg_res.get("status")
                    f.write(f"--- Segment {segment_index} ({start_time:.2f}s - {end_time:.2f}s) ---\n")
                    f.write(f"Status: {status}\n")
                    transcribed_segments = seg_res.get("transcription_segments", [])
                    if transcribed_segments:
                        transcribed_segments.sort(key=lambda x: x.get('start', 0))
                        f.write("Transcription:\n")
                        for ts_seg in transcribed_segments:
                            speaker = ts_seg.get('speaker', 'UNK')
                            seg_start = ts_seg.get('start', 0)
                            seg_end = ts_seg.get('end', 0)
                            text = ts_seg.get('text', '').strip() or "(Empty transcription)"
                            f.write(f"  [{seg_start:.2f}s - {seg_end:.2f}s] {speaker}: {text}\n")
                    else:
                        f.write("Transcription: (Not available or failed)\n")
                    f.write("\n" + "-"*20 + "\n\n")
            elif results.get("transcription") and isinstance(results["transcription"], dict) and "text" in results["transcription"]:
                f.write("Full Transcription:\n")
                f.write(results["transcription"]["text"])
            else:
                f.write("(No transcription available)")
        audio_obj.add_provenance('write_outputs', {'master_transcript_path': master_transcript_path})
        logger.info(f"Saved master transcript to {master_transcript_path}")
    except Exception as e:
        logger.error(f"Failed to generate master transcript: {e}", exc_info=True)

    # --- Write Results YAML ---
    results_path = os.path.join(output_dir, "results.yaml")
    try:
        cleaned_results = results.copy()
        if "segments" in cleaned_results:
            segment_summaries = []
            for seg_res in results.get("segments", []):
                summary = {
                    "segment_index": seg_res.get("segment_index"),
                    "status": seg_res.get("status"),
                    "start": seg_res.get("start_original"),
                    "end": seg_res.get("end_original"),
                    "duration": seg_res.get("duration"),
                    "num_transcribed_chunks": len(seg_res.get("transcription_segments", [])),
                    "output_dir": seg_res.get("segment_output_dir")
                }
                segment_summaries.append(summary)
            cleaned_results["segments"] = segment_summaries
        if "transcription" in cleaned_results:
            if isinstance(cleaned_results["transcription"], dict) and "text" in cleaned_results["transcription"]:
                cleaned_results["transcription_summary"] = cleaned_results["transcription"]["text"][:200] + "..."
            del cleaned_results["transcription"]
        with open(results_path, "w", encoding='utf-8') as f:
            yaml.dump(cleaned_results, f, default_flow_style=False, sort_keys=False, width=120)
        audio_obj.add_provenance('write_outputs', {'results_yaml_path': results_path})
        logger.info(f"Saved final results YAML to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save final results YAML: {e}", exc_info=True)
    return audio_obj 