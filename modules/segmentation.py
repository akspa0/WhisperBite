import os
import logging
from pydub import AudioSegment
import json

def segment_audio(audio_obj, config=None):
    """
    Segments audio by pairing each 'speech' or 'conversation' event (after a 'ringing', or at the start) with the next 'hang-up tones' event.
    Saves segments to config['stage_dirs']['segments'] and returns updated audio_obj with provenance and segment paths.
    Also saves a summary of segment boundaries and event triggers to a JSON file in the segments folder.
    If no valid segments are found, creates a fallback segment covering the whole file for debugging.
    """
    logger = logging.getLogger(__name__)
    audio_path = audio_obj.path
    all_events = audio_obj.metadata.get('clap_events', {})
    logger.info(f"CLAP event types present: {list(all_events.keys())}")
    output_dir = config.get('stage_dirs', {}).get('segments', config.get('output_dir', '.'))
    os.makedirs(output_dir, exist_ok=True)
    min_duration_s = config.get('min_duration_s', 5.0)
    padding_ms = config.get('padding_ms', 500)
    # Flatten and sort all events
    event_list = []
    for type_key, events_list in all_events.items():
        for event in events_list:
            event_data = event.copy()
            event_data['type'] = type_key
            event_list.append(event_data)
    event_list.sort(key=lambda x: x['start'])
    if not event_list:
        logger.warning("No events provided for segmentation.")
        audio_obj.add_provenance('segment_audio', {'segments': []})
        return audio_obj
    # Build lists of each event type
    ringing_events = [e for e in event_list if e['type'] == 'telephone ringing']
    speech_events = [e for e in event_list if e['type'] in ('speech', 'conversation')]
    hangup_events = [e for e in event_list if e['type'] == 'hang-up tones']
    # Main pairing logic
    saved_segments = []
    segment_summaries = []
    segment_index = 0
    try:
        audio = AudioSegment.from_file(audio_path)
        audio_duration_ms = len(audio)
        # Start at the first speech/conversation event after a ringing, or at the very beginning
        i = 0
        last_hangup_end = 0.0
        while i < len(speech_events):
            speech_event = speech_events[i]
            # Find the most recent ringing before this speech (optional, for logging)
            prev_ringing = None
            for r in reversed(ringing_events):
                if r['start'] < speech_event['start']:
                    prev_ringing = r
                    break
            # Find the next hang-up after this speech
            next_hangup = None
            for h in hangup_events:
                if h['start'] > speech_event['start']:
                    next_hangup = h
                    break
            if not next_hangup:
                break  # No more hang-ups, stop
            # Only create a segment if this speech starts after the last hang-up (no overlap)
            if speech_event['start'] < last_hangup_end:
                i += 1
                continue
            cut_start_ms = max(0, int(speech_event['start'] * 1000) - padding_ms)
            cut_end_ms = min(audio_duration_ms, int(next_hangup['end'] * 1000) + padding_ms)
            segment_duration_s = (cut_end_ms - cut_start_ms) / 1000.0
            if segment_duration_s >= min_duration_s:
                segment_audio = audio[cut_start_ms:cut_end_ms]
                segment_filename = f"{segment_index:04d}_conv_{int(cut_start_ms/1000.0)}_{int(cut_end_ms/1000.0)}.wav"
                segment_path = os.path.join(output_dir, segment_filename)
                try:
                    segment_audio.export(segment_path, format="wav")
                except Exception as export_err:
                    logger.error(f"Failed to export segment {segment_index}: {export_err}")
                    i += 1
                    continue
                logger.info(f"Saved segment {segment_index}: {segment_filename} [{cut_start_ms/1000.0:.2f}s - {cut_end_ms/1000.0:.2f}s], speech={speech_event['start']:.2f}s, hangup={next_hangup['start']:.2f}s, prev_ringing={prev_ringing['start'] if prev_ringing else None}")
                saved_segments.append({
                    'path': segment_path,
                    'start': cut_start_ms / 1000.0,
                    'end': cut_end_ms / 1000.0,
                    'duration': segment_duration_s,
                    'speech_event': speech_event,
                    'hangup_event': next_hangup,
                    'prev_ringing': prev_ringing
                })
                segment_summaries.append({
                    'segment_index': segment_index,
                    'filename': segment_filename,
                    'start_s': cut_start_ms / 1000.0,
                    'end_s': cut_end_ms / 1000.0,
                    'duration_s': segment_duration_s,
                    'speech_event': speech_event,
                    'hangup_event': next_hangup,
                    'prev_ringing': prev_ringing
                })
                segment_index += 1
                last_hangup_end = next_hangup['end']
            i += 1
        # Fallback: if no segments found, create a segment covering the whole file
        if not saved_segments:
            logger.warning("No valid segments found. Creating fallback segment covering the whole file.")
            segment_audio = audio
            segment_filename = "0000_fullfile_fallback.wav"
            segment_path = os.path.join(output_dir, segment_filename)
            try:
                segment_audio.export(segment_path, format="wav")
            except Exception as export_err:
                logger.error(f"Failed to export fallback segment: {export_err}")
            saved_segments.append({
                'path': segment_path,
                'start': 0.0,
                'end': audio_duration_ms / 1000.0,
                'duration': audio_duration_ms / 1000.0,
                'speech_event': None,
                'hangup_event': None,
                'prev_ringing': None,
                'fallback': True
            })
            segment_summaries.append({
                'segment_index': 0,
                'filename': segment_filename,
                'start_s': 0.0,
                'end_s': audio_duration_ms / 1000.0,
                'duration_s': audio_duration_ms / 1000.0,
                'speech_event': None,
                'hangup_event': None,
                'prev_ringing': None,
                'fallback': True
            })
    except Exception as e:
        logger.error(f"Error during segmentation: {e}", exc_info=True)
        audio_obj.add_provenance('segment_audio', {'segments': []})
        return audio_obj
    logger.info(f"Finished segmentation. Saved {len(saved_segments)} segments.")
    # Save segment summary to JSON
    summary_path = os.path.join(output_dir, 'segment_summary.json')
    try:
        with open(summary_path, 'w') as f:
            json.dump(segment_summaries, f, indent=2)
        logger.info(f"Segment summary written to {summary_path}")
    except Exception as e:
        logger.warning(f"Failed to write segment summary: {e}")
    audio_obj.add_provenance('segment_audio', {'segments': saved_segments})
    return audio_obj 