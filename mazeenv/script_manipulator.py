# Modified from https://github.com/microsoft/MazeExplorer/blob/e66a2d405e08bc75e51bd38a2b96959c554fe773/mazeexplorer/script_manipulator.py

from string import Template
import os
import random

dir_path = os.path.dirname(os.path.realpath(__file__))


def write_config(wad, actions, episode_timeout, living_reward, resolution):
    """
    args:
    wad: (str) name of corresponding wad file
    actions: (str) list of available actions (default: "MOVE_FORWARD TURN_LEFT TURN_RIGHT")
    """
    # open the file
    filein = open(os.path.join(dir_path, 'config_template.txt'))
    # read it
    src = Template(filein.read())

    mission_wad = os.path.splitext(os.path.basename(wad))[0]
    d = {'actions': actions, 'mission_wad': mission_wad, 'episode_timeout': episode_timeout,
         'living_reward': living_reward, 'resolution': resolution}

    # do the substitution
    result = src.substitute(d)

    f = open(wad + ".cfg", "w+")
    f.write(result)
    f.close()

    return wad + ".cfg"


def sample_key_textures(categories):
    key_texture_list = []
    for category in categories.values():
        key_texture_list.append(str(random.randrange(len(category))))
    return key_texture_list


def format_key_textures(categories):
    max_category_size = -1
    for category in categories.values():
        max_category_size = max(max_category_size, len(category))
    ret = '{'
    for cat_idx, category in enumerate(categories.values()):
        ret += '{'
        for text_idx in range(max_category_size):
            if text_idx <= len(category) - 1:
                ret += '"{}"'.format(category[text_idx])
            else:
                ret += '"NONE"'
            if text_idx != max_category_size - 1:
                ret += ','
        ret += '}'
        if cat_idx != len(categories) - 1:
            ret += ','
    ret += '}'
    return ret, max_category_size


def generate_boxes(boxes_dims, xmin, xmax, ymin, ymax):
    key_boxes_x = []
    key_boxes_y = []

    num_boxes_x, num_boxes_y = boxes_dims
    x_start = xmin;
    y_start = ymin;
    for y_idx in range(num_boxes_y):
        if y_idx == num_boxes_y - 1:
            y_end = ymax
        else:
            y_end = y_start + (ymax - ymin) / num_boxes_y
        
        x_start = xmin;
        for x_idx in range(num_boxes_x):
            if x_idx == num_boxes_x - 1:
                x_end = xmax
            else:
                x_end = x_start + (xmax - xmin) / num_boxes_x
            key_boxes_x.append((x_start, x_end))
            key_boxes_y.append((y_start, y_end))
            x_start = x_end
        y_start = y_end

    key_boxes_x_str = '{' + ','.join([('{%.4f, %.4f}' % (b[0], b[1])) for \
                                      b in key_boxes_x]) + '}'
    key_boxes_y_str = '{' + ','.join([('{%.4f, %.4f}' % (b[0], b[1])) for \
                                      b in key_boxes_y]) + '}'
    num_boxes = num_boxes_x * num_boxes_y

    return num_boxes, key_boxes_x, key_boxes_y, key_boxes_x_str, key_boxes_y_str


def write_acs(random_player_spawn, random_textures, random_key_positions, map_size, number_maps,
              floor_texture, ceiling_texture, wall_texture, key_categories,
              random_key_textures, default_spawn_pos, default_key_pos, acs_path,
              texture_list, keys_used_list, use_key_boxes, shuffle_obj_pos,
              boxes_dims, random_player_spawn_angle,
              def_player_spawn_angle=None, seed=None):
    """
    args:
    random_player_spawn: (bool) whether or not agent should be randomly placed in maze at spawn time
    random_textures: (bool) whether or not textures (walls, floors etc.) should be randomised.
    """
    BLOCK_SIZE = 96

    xmin = float(BLOCK_SIZE / 2)
    ymin = float(BLOCK_SIZE / 2)
    xmax = float(BLOCK_SIZE / 2 + 2 * round(map_size[0] / 2) * BLOCK_SIZE)
    ymax = float(BLOCK_SIZE / 2 + 2 * round(map_size[1] / 2) * BLOCK_SIZE)

    if seed:
        random.seed(seed)

    maze_acs = os.path.join(dir_path, 'maze.acs')
    if os.path.exists(maze_acs):
        os.remove(maze_acs)

    # open the file
    filein = open(os.path.join(dir_path, acs_path))
    # read it
    src = Template(filein.read())

    doom_textures = '{"' + '","'.join(texture_list) + '"};'
    num_textures = len(texture_list)

    num_keys = len(key_categories)

    sampled_key_textures = sample_key_textures(key_categories)
    default_key_textures = '{' + ','.join(sampled_key_textures) + '}'
    
    key_textures, max_category_size = format_key_textures(key_categories)
    num_key_textures = '{' + ','.join([str(len(c)) for c in key_categories.values()]) + '}'

    keys_used = '{' + ','.join([str(k) for k in keys_used_list]) + '}'
    num_keys_used = len(keys_used_list)

    if use_key_boxes:
        num_boxes, key_boxes_x, key_boxes_y, key_boxes_x_str, key_boxes_y_str = \
            generate_boxes(boxes_dims, xmin, xmax, ymin, ymax)
    else:
        # placeholder values
        num_boxes = 1
        key_boxes_x = None
        key_boxes_y = None
        key_boxes_x_str = '{{0.0,0.0}}'
        key_boxes_y_str = '{{0.0,0.0}}'


    if default_key_pos is not None:
        abs_keys_spawn_x = ', '.join([('%.4f' % pos[0]) for pos in default_key_pos])
        abs_keys_spawn_y = ', '.join([('%.4f' % pos[1]) for pos in default_key_pos])
    elif use_key_boxes:
        box_samples = random.sample(range(num_boxes), num_keys)
        keys_x = []
        keys_y = []
        for box_idx in box_samples:
            box_x = key_boxes_x[box_idx]
            keys_x.append(random.uniform(box_x[0], box_x[1]))
            box_y = key_boxes_y[box_idx]
            keys_y.append(random.uniform(box_y[0], box_y[1]))
        abs_keys_spawn_x = ', '.join([('%.4f' % x) for x in keys_x])
        abs_keys_spawn_y = ', '.join([('%.4f' % y) for y in keys_y])
    else:
        abs_keys_spawn_x = ', '.join([str(random.uniform(xmin, xmax)) \
            for _ in range(num_keys)])
        abs_keys_spawn_y = ', '.join([str(random.uniform(ymin, ymax)) \
            for _ in range(num_keys)])

    if default_spawn_pos is not None:
        abs_spawn_x = '%.4f' % default_spawn_pos[0]
        abs_spawn_y = '%.4f' % default_spawn_pos[1]
    else:
        abs_spawn_x = '%.4f' % (random.uniform(xmin, xmax))
        abs_spawn_y = '%.4f' % (random.uniform(ymin, ymax))

    if def_player_spawn_angle is not None:
        spawn_angle = '%.2f' % def_player_spawn_angle
    else:
        spawn_angle = '%.2f' % (random.random())

    d = {
        'num_keys': num_keys, 'random_spawn': random_player_spawn, 'random_textures': random_textures,
        'textures': doom_textures, 'num_textures': num_textures,
        'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
        'floor_texture': floor_texture, 'ceiling_texture': ceiling_texture, 'wall_texture': wall_texture,
        'random_key_positions': random_key_positions, 'spawn_angle': spawn_angle, 'number_keys_maps': num_keys,
        'random_key_textures': random_key_textures, 'num_key_textures': num_key_textures,
        'max_category_size': max_category_size, 'key_textures': key_textures,
        'default_key_textures': default_key_textures,
        'abs_keys_spawn_x': abs_keys_spawn_x, 'abs_keys_spawn_y': abs_keys_spawn_y,
        'abs_spawn_x': abs_spawn_x, 'abs_spawn_y': abs_spawn_y,
        'num_keys_used': num_keys_used, 'keys_used': keys_used,
        'random_spawn_angle': random_player_spawn_angle,
        'shuffle_obj_pos': shuffle_obj_pos,
        'spawn_angle': spawn_angle,
        'use_key_boxes': use_key_boxes,
        'num_boxes': num_boxes, 'key_boxes_x': key_boxes_x_str, 'key_boxes_y': key_boxes_y_str,
    }

    # do the substitution
    result = src.substitute(d)

    f = open(os.path.join(dir_path, "maze.acs"), "w+")
    f.write(result)
    f.close()
