import os

def make_rotations(level_lines):
    tar_loc, box_loc, alttar_loc, altbox_loc, path1_loc, path2_loc = level_lines[0]
    rotate_lines_1 = [((tar_loc[1], 7-tar_loc[0]), (box_loc[1], 7-box_loc[0]), (alttar_loc[1], 7-alttar_loc[0]), (altbox_loc[1], 7-altbox_loc[0]), (path1_loc[1], 7-path1_loc[0]), (path2_loc[1], 7-path2_loc[0]))]
    rotate_lines_2 = [((7-tar_loc[0], 7-tar_loc[1]), (7-box_loc[0], 7-box_loc[1]), (7-alttar_loc[1], 7-alttar_loc[0]), (7-altbox_loc[1], 7-altbox_loc[0]), (7-path1_loc[1], 7-path1_loc[0]), (7-path2_loc[1], 7-path2_loc[0]))]
    rotate_lines_3 = [((7-tar_loc[1], tar_loc[0]), (7-box_loc[1], box_loc[0]), (7-alttar_loc[1], alttar_loc[0]), (7-altbox_loc[1], altbox_loc[0]), (7-path1_loc[1], path1_loc[0]), (7-path2_loc[1], path2_loc[0]))]
    for y in range(10):
        new_line_1 = []
        new_line_2 = []
        new_line_3 = []
        for x in range(10):
            new_line_1.append(level_lines[1:][-(x+1)][y])
            new_line_2.append(level_lines[1:][-(y+1)][-(x+1)])
            new_line_3.append(level_lines[1:][x][-(y+1)])
        rotate_lines_1.append(new_line_1)
        rotate_lines_2.append(new_line_2)
        rotate_lines_3.append(new_line_3)
    return (level_lines, rotate_lines_1, rotate_lines_2, rotate_lines_3)

def process_levels(levels):
    all_levels = []
    for j in range(len(levels.split(";")[1:])):

        level_info = levels.split(";")[1:][j].split("\n")[0].split("-")

        tar_loc = tuple([int(c) for c in level_info[1].split(",")])
        box_loc = tuple([int(c) for c in level_info[2].split(",")])
        alttar_loc = tuple([int(c) for c in level_info[3].split(",")])
        altbox_loc = tuple([int(c) for c in level_info[4].split(",")])
        path1_loc = tuple([int(c) for c in level_info[5].split(",")])
        path2_loc = tuple([int(c) for c in level_info[6].split(",")])

        mirror_tar_loc = tuple([tar_loc[0], 7-tar_loc[1]]) # assume reflection is in vertical axis
        mirror_box_loc = tuple([box_loc[0], 7-box_loc[1]])
        mirror_alttar_loc = tuple([alttar_loc[0], 7-alttar_loc[1]])
        mirror_altbox_loc = tuple([altbox_loc[0], 7-altbox_loc[1]])
        mirror_path1_loc = tuple([path1_loc[0], 7-path1_loc[1]])
        mirror_path2_loc = tuple([path2_loc[0], 7-path2_loc[1]])
    
        level_strings = levels.split(";")[1:][j].split("\n")[1:]
        if level_strings[-1] == "":
            level_strings = level_strings[:-1]
        raw_lines, mirror_lines = [(tar_loc, box_loc, alttar_loc, altbox_loc, path1_loc, path2_loc)], [(mirror_tar_loc, mirror_box_loc, mirror_alttar_loc, mirror_altbox_loc, mirror_path1_loc, mirror_path2_loc)]
        for line in level_strings:
            line_list = list(line)
            mirror_list = []
            for i in range(len(line)):
                mirror_list.append(line_list[-(i+1)])
            raw_lines.append(line_list)
            mirror_lines.append(mirror_list)
        all_levels += make_rotations(raw_lines)
        all_levels += make_rotations(mirror_lines)
    return all_levels

if __name__=="__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="create clean+corrupted experiment Sokoban levels")
    parser.add_argument("--expname", type=str, required=True)
    args = parser.parse_args()

    info_dict = {}

    expname = args.expname
    print(f"Creating clean+corrupted Sokoban levels for {expname}")

    with open(f"./exp-levels-txt/{expname}/clean.txt") as f:
        clean_levels = f.read()
    with open(f"./exp-levels-txt/{expname}/corrupt.txt") as f:
        corrupt_levels = f.read()

    all_clean_levels = process_levels(clean_levels)
    all_corrupt_levels = process_levels(corrupt_levels)

    level_id = 0
    exp_dir = f"./boxoban-levels/experiments/{expname}"
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    for clean_level, corrupt_level in zip(all_clean_levels, all_corrupt_levels):

        clean_info, corrupt_info = clean_level[0], corrupt_level[0]
        info_dict[f"{expname}_{level_id:04}"] = {"tar_loc": list(clean_info[0]), 
                                             "box_loc": list(clean_info[1]), 
                                             "alttar_loc": list(clean_info[2]),
                                             "altbox_loc": list(clean_info[3]),
                                              "path1_loc": list(clean_info[4]),
                                              "path2_loc": list(clean_info[5])}

        level_dir = exp_dir + f"/{level_id:04}"
        if not os.path.exists(level_dir):
            os.mkdir(level_dir)
            os.mkdir(level_dir+"/clean")
            os.mkdir(level_dir+"/corrupt")
        clean_level = ["".join(line) for line in clean_level]
        clean_level = [f"; {level_id}"] + clean_level
        clean_level = "\n".join(clean_level)
        with open(level_dir+"/clean/000.txt", "w") as f:
            f.write(clean_level)
        corrupt_level = ["".join(line) for line in corrupt_level]
        corrupt_level = [f"; {level_id}"] + corrupt_level
        corrupt_level = "\n".join(corrupt_level)
        with open(level_dir+"/corrupt/000.txt", "w") as f:
            f.write(corrupt_level)
        level_id += 1

    info_df = pd.DataFrame(info_dict).to_csv(f"./exp-levels-txt/{expname}.csv")