#pragma once
#include "hitablelist.h"

hitable* random_scene();

hitable* simple_light();

hitable* cornell_box();

hitable* cube_light();

__global__ void spot(hitable_list* list);