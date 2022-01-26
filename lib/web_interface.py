from maps.models import Task
from lib.pipeline import call_graphviz, get_graphviz_map, call_graphviz_scale, set_status, call_hmds
from re import sub, search
import time
from datetime import datetime
from time import strftime

import logging
log = logging.getLogger('gmap_web')

def create_task(task_parameters, user_ip):
	# set up new object

	task = Task()
	task.creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	task.creation_ip = user_ip

	task.input_dot = task_parameters['dotfile']
	task.vis_type = "Gmap"
	task.layout_algorithm = task_parameters['layout_algorithm']
	task.iterations = task_parameters['iterations']
	task.opt_alpha = task_parameters.get('opt_alpha', 'false')
	task.hyperbolic_projection = task_parameters.get('hyperbolic','false')
	task.color_scheme = "blue" #task_parameters['color_scheme']
	task.convergence = task_parameters.get('convergence', 'false')
	task.status = 'created'

	task.save()

	return task

def create_map(task, *args):
	# set up new objects

	dot_rep = call_graphviz(task)
	svg_rep = 2
	if dot_rep is None or svg_rep is None:
		return

	width = 2
	height = 2
	#svg_rep, width, height = strip_dimensions(svg_rep.decode())

	task.dot_rep = dot_rep
	task.svg_rep = svg_rep
	task.width = width
	task.height = height
	task.status = 'completed'
	task.save()

def get_formatted_map(task, format):
	return get_graphviz_map(task.dot_rep, format)

def strip_dimensions(svg):
    """having width and height attributes as well as a viewbox will cause openlayers to not display the svg propery, so we strip those attributes out"""
    svg = sub('<title>%3</title>', '', svg, count=1)

    match_re = '<svg width="(.*)pt" height="(.*)pt"'
    replacement = '<svg'
    try:
        width, height = map(float, search(match_re, svg).groups())
    except Exception:
        width, height = 0.0, 0.0
    return sub(match_re, replacement, svg, count=1), width, height
