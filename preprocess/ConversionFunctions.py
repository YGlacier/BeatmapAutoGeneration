def beatmap_to_training_data(beatmap_path, training_data_path):
    with open(beatmap_path, "r") as beatmap_file:
        lines = beatmap_file.read().splitlines()

    # find audio file name
    i = lines.index("[General]")
    audio_filename = lines[i+1].split(": ")[1]
    
    # find beat length
    i = lines.index("[TimingPoints]")
    beat_length = lines[i+1].split(",")[1]

    # strim hit object expression
    timing_list = []
    action_list = []
    column_list = []
    i = lines.index("[HitObjects]") + 1
    while i < len(lines):
        split_line = lines[i].split(",")
        # colume
        if int(split_line[0]) == 64:
            column = 0
        elif int(split_line[0]) == 192:
            column = 1
        elif int(split_line[0]) == 320:
            column = 2
        elif int(split_line[0]) == 448:
            column = 3

        # timing
        timing = int(split_line[2])

        # action
        # hit
        if int(split_line[3]) != 128:
            action = 1
        # hold
        else:
            action = 2
            release_timing = int(split_line[5].split(":")[0])
            release_action = 3

        # append to lists
        if action == 1:
            timing_list.append(timing)
            action_list.append(action)
            column_list.append(column)
        else:
            timing_list.append(timing)
            action_list.append(action)
            column_list.append(column)

            timing_list.append(release_timing)
            action_list.append(release_action)
            column_list.append(column)

        i += 1

    # write to .dat file
    with open(training_data_path, "w") as training_data_file:
        training_data_file.write(audio_filename + "\n")
        i = 0 
        while i < len(timing_list):
            training_data_file.write(str(timing_list[i]) + "," + str(action_list[i]) + "," + str(column_list[i]) + "\n")
            i += 1
        
def generated_data_to_beatmap(generated_data_path, beatmap_path):
    with open(generated_data_path, "r") as generated_file:
        lines = generated_file.read().splitlines()

    beatmap_file = open(beatmap_path, "w")
    beatmap_file.write("osu file format v14\n\n")
    beatmap_file.write("[General]\n")
    beatmap_file.write("AudioFilename: " + lines[0] + "\n")
    beatmap_file.write("Mode: 3\n\n")
    beatmap_file.write("[Metadata]\n")
    beatmap_file.write("Title:" + lines[0] + "\n")
    beatmap_file.write("TitleUnicode:" + lines[0] + "\n")
    beatmap_file.write("Version:Generated\n\n")
    beatmap_file.write("[Difficulty]\n")
    beatmap_file.write("HPDrainRate:7\n")
    beatmap_file.write("CircleSize:4\n")
    beatmap_file.write("OverallDifficulty:7\n")
    beatmap_file.write("ApproachRate:5.5\n")
    beatmap_file.write("SliderMultiplier:2\n")
    beatmap_file.write("SliderTickRate:1\n\n")
    beatmap_file.write("[Events]\n\n")
    beatmap_file.write("[TimingPoints]\n")
    beatmap_file.write("0,600,4,2,1,100,1,0\n\n")
    beatmap_file.write("[Colours]\n\n")
    beatmap_file.write("[HitObjects]\n")

    i = 1
    while i < len(lines):
        split_line = lines[i].split(",")
        if int(split_line[1]) == 1:
            beatmap_file.write(str(int(split_line[2]) * 128 + 64) + ",192," + split_line[0] + ",1,0,0:0:0:0:\n")
            i += 1
        elif int(split_line[1]) == 2:
            beatmap_file.write(str(int(split_line[2]) * 128 + 64) + ",192," + split_line[0] + ",128,0," + lines[i+1].split(",")[0] + ":0:0:0:0:\n")
            i += 2





generated_data_to_beatmap("../LocalTest/TestBeatMap.dat", "../LocalTest/Reconstruct.osu")