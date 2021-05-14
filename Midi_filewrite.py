import midi
# file = "test.mid"

# pattern = midi.Pattern(resolution=960) #このパターンがmidiファイルに対応しています。

# track = midi.Track() #トラックを作ります
# pattern.append(track) #パターンに作ったトラックを追加します。

# ev = midi.SetTempoEvent(tick=0, bpm=120) #テンポを設定するイベントを作ります
# track.append(ev) #イベントをトラックに追加します。

# e = midi.NoteOnEvent(tick=0, velocity=100, pitch=midi.G_4) #ソの音を鳴らし始めるイベントを作ります。
# track.append(e)

# e = midi.NoteOffEvent(tick=960, velocity=100, pitch=midi.G_4) #ソの音を鳴らし終えるイベントを作ります。
# track.append(e)

# eot = midi.EndOfTrackEvent(tick=1) #トラックを終えるイベントを作ります
# track.append(eot)

# midi.write_midifile(file, pattern) #パターンをファイルに書き込みます。

import pretty_midi
# Create a PrettyMIDI object
cello_c_chord = pretty_midi.PrettyMIDI()
# Create an Instrument instance for a cello instrument
cello_program = pretty_midi.instrument_name_to_program('Trumpet')
cello = pretty_midi.Instrument(program=cello_program)

# Iterate over note names, which will be converted to note number later
for note_name in ['C5', 'E5', 'G5']:
    # Retrieve the MIDI note number for this note name
    note_number = pretty_midi.note_name_to_number(note_name)
    # Create a Note instance, starting at 0s and ending at .5s
    note = pretty_midi.Note(
        velocity=100, pitch=note_number, start=0, end=.5)
    # Add it to our cello instrument
    cello.notes.append(note)
# Add the cello instrument to the PrettyMIDI object
cello_c_chord.instruments.append(cello)
# Write out the MIDI data
cello_c_chord.write('Viola-C-chord.mid')
print(cello_c_chord.synthesize()) 