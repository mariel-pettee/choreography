import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import juggle_axes
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
import matplotlib

# ask matplotlib to plot up to 2^128 frames in animations
matplotlib.rcParams['animation.embed_limit'] = 2**128

def update_points(time, points, df, params):
  '''
  Callback function called by plotting function below. Mutates the vertex
  positions of each value in `points` so the animation moves
  @param int time: the index of the time slice to visualize within `df`
  @param mpl_toolkits.mplot3d.art3d.Path3DCollection points: geometry to mutate
  @param numpy.ndarray df: a 2D numpy array with shape d[vert][time][dimension]
  '''
  points._offsets3d = juggle_axes(df[:,time,0], df[:,time,1], df[:,time,2], 'z')

def update_lines(time, lines, df, params):
  '''
  Callback function called by plotting function below. Mutates the vertex
  positions of each value in `points` so the animation moves
  @param int time: the index of the time slice to visualize within `df`
  @param mpl_toolkits.mplot3d.art3d.Path3DCollection lines: geometry to mutate
  @param numpy.ndarray df: a 2D numpy array with shape d[vert][time][dimension]
  '''
  new_positions = [ [df[i,time,:], df[j,time,:]] for i, j in params['edges'] ]
  for idx, line in enumerate(lines):
    a, b = new_positions[idx]
    line[0].set_data([a[0], b[0]], [a[1], b[1]])
    line[0].set_3d_properties([a[2], b[2]])
  return lines

def animate(df, edges=[], axes=None, frames=50, speed=45, figsize=(7,5), colors=None):
  '''
  General function that can plot numpy arrays in either of two shapes.
  @param numpy.ndarray df: a 2D numpy array with shape d[vert][time][dimension]
  @param numpy.ndarray edges: 2 2D numpy array with shape [[i, j]] where i and j
    are indices into the `vertices` in X
  @param dict axes: dict that maps {x, y, z} keys to (min_axis_val, max_axis_val)
  @param int frames: the number of time slices to animate
  @param int speed: the temporal duration of each frame. Increase to boost fps
  @param tuple figsize: the size of the figure to render
  @param {str|list}: string or list of  color values (if list, one per edge to be drawn)
  '''
  if axes is None:
    axes = {'x': (0,1), 'y': (0,1), 'z': (0, 1.5)}
  fig = plt.figure(figsize=figsize)
  ax = p3.Axes3D(fig)
  ax.set_xlim(*axes['x'])
  ax.set_ylim(*axes['y'])
  ax.set_zlim(*axes['z'])
  plt.close(fig)
  if edges:
    params = {'edges': edges}
    callback = update_lines
    lines, geoms = [ [df[i,0,:], df[j,0,:]] for i, j in edges ], []
    for idx, i in enumerate(lines):
      if colors and isinstance(colors, list): c = colors[idx]
      elif colors: c = colors
      else: c = plt.cm.RdYlBu(idx/len(lines))
      geoms.append( ax.plot([i[0][0], i[1][0]], [i[0][1], i[1][1]], [i[0][2], i[1][2]], color=c) )
  else:
    params = None
    callback = update_points
    geoms = ax.scatter(df[:,0,0], df[:,0,1], df[:,0,2], depthshade=False) # x,y,z vals
  return animation.FuncAnimation(fig,
    callback,
    frames,
    interval=speed,
    fargs=(geoms, df, params),
    blit=False,
  ).to_html5_video()


def plot_labelled_points(x=None, frame_idx=0, figsize=(14,10), font_size=8, axes=None, text=True):
  '''
  Plot the vertices from a body with index numbers next to them (useful for cherry-
  picking vertices to keep in a model)
  @param numpy.ndarray x: a 3D numpy array with shape X[vert][time][dimension]
  @param int frame_idx: the index position of the frame to draw
  @param dict axes: dict that maps {x, y, z} keys to (min_axis_val, max_axis_val)
  @param tuple figsize: the size of the figure to render
  @param bool text: whether or not to label the vertices with text
  '''
  if axes is None:
    axes = {'x': (0,1), 'y': (0,1), 'z': (0, 1.5)}
  fig = plt.figure(figsize=figsize)
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlim(*axes['x'])
  ax.set_ylim(*axes['y'])
  ax.set_zlim(*axes['z'])
  m = x[:,frame_idx,:].squeeze() # m is an array of (x,y,z) coordinate triplets
  for i in range(len(m)): # plot each point + it's index as text above
    if text:
      ax.text(m[i,0], m[i,1], m[i,2], '%s' % (labels[i]), size=font_size, zorder=1, color='k')
    else:
      ax.scatter(m[i,0], m[i,1], m[i,2], color='b')
      ax.text(m[i,0], m[i,1], m[i,2], '%s' % (i), size=font_size, zorder=1, color='k')

def plot_lines(x=None, time_idx=0, edges=[], axes=None, colors=None, figsize=(7,5)):
  '''
  Plot edges between a collection of vertex pairs
  @param numpy.ndarray x: a 3D numpy array with shape [vertex][time][dimension]
  @param numpy.ndarray edges: 2 2D numpy array with shape [[i, j]] where i and j
    are indices into the `vertices` in `x`
  @param int time_idx: the time slice for which we will draw lines
  @param dict axes: dict that maps {x, y, z} keys to (min_axis_val, max_axis_val)
  @param list colors: list of color values, one per edge to be drawn
  @param tuple figsize: the size of the figure to render
  '''
  if axes is None:
    axes = {'x': (0,1), 'y': (0,1), 'z': (0, 1.5)}
  fig = plt.figure(figsize=figsize)
  ax = p3.Axes3D(fig)
  ax.set_xlim(*axes['x'])
  ax.set_ylim(*axes['y'])
  ax.set_zlim(*axes['z'])
  lines = [ [X[i,time_idx,:], X[j,time_idx,:]] for i, j in edges ]
  for idx, i in enumerate(lines):
    c = colors[idx] if colors else plt.cm.RdYlBu(idx/len(lines))
    ax.plot([i[0][0], i[1][0]], [i[0][1], i[1][1]], [i[0][2], i[1][2]], color=c)
    
def update_lines_ghost(time, lines_real, df_real, lines_pred, df_pred, params, ghost_shift):
  '''
  Callback function called by plotting function below. Mutates the vertex
  positions of each value in `points` so the animation moves
  @param int time: the index of the time slice to visualize within `df`
  @param mpl_toolkits.mplot3d.art3d.Path3DCollection lines: geometry to mutate
  @param numpy.ndarray df: a 2D numpy array with shape d[vert][time][dimension]
  '''
  new_positions_real = [ [df_real[i,time,:], df_real[j,time,:]] for i, j in params['edges'] ]
  for idx, line in enumerate(lines_real):
    a, b = new_positions_real[idx]
    line[0].set_data([a[0], b[0]], [a[1], b[1]])
    line[0].set_3d_properties([a[2], b[2]])
    
  new_positions_pred = [ [df_pred[i,time,:], df_pred[j,time,:]] for i, j in params['edges'] ]
  for idx, line in enumerate(lines_pred):
    a, b = new_positions_pred[idx]
    line[0].set_data([a[0]-ghost_shift, b[0]-ghost_shift], [a[1], b[1]])
    line[0].set_3d_properties([a[2], b[2]])
  return lines_pred

def animate_ghost(df_real, df_pred, edges=[], axes=None, frames=50, speed=45, figsize=(7,5), colors=None, ghost_shift=0.3):
  '''
  General function that can plot numpy arrays in either of two shapes.
  @param numpy.ndarray df: a 2D numpy array with shape d[vert][time][dimension]
  @param numpy.ndarray edges: 2 2D numpy array with shape [[i, j]] where i and j
    are indices into the `vertices` in X
  @param dict axes: dict that maps {x, y, z} keys to (min_axis_val, max_axis_val)
  @param int frames: the number of time slices to animate
  @param int speed: the temporal duration of each frame. Increase to boost fps
  @param tuple figsize: the size of the figure to render
  @param {str|list}: string or list of  color values (if list, one per edge to be drawn)
  '''
  if axes is None:
    axes = {'x': (0,1), 'y': (0,1), 'z': (0, 1.5)}
  fig = plt.figure(figsize=figsize)
  ax = p3.Axes3D(fig)
  ax.set_xlim(*axes['x'])
  ax.set_ylim(*axes['y'])
  ax.set_zlim(*axes['z'])
  plt.close(fig)
  if edges:
    params = {'edges': edges}
    callback = update_lines_ghost
    lines_real, geoms_real = [ [df_real[i,0,:], df_real[j,0,:]] for i, j in edges ], []
    for idx, i in enumerate(lines_real):
        c = 'black'
#       if colors and isinstance(colors, list): c = colors[idx]
#       elif colors: c = colors
#       else: c = plt.cm.RdYlBu(idx/len(lines_real))
        geoms_real.append( ax.plot([i[0][0], i[1][0]], [i[0][1], i[1][1]], [i[0][2], i[1][2]], color=c) )
    lines_pred, geoms_pred = [ [df_pred[i,0,:], df_pred[j,0,:]] for i, j in edges ], []
    for idx, i in enumerate(lines_pred):
      if colors and isinstance(colors, list): c = colors[idx]
      elif colors: c = colors
      else: c = plt.cm.RdYlBu(idx/len(lines_pred))
      geoms_pred.append( ax.plot([i[0][0], i[1][0]], [i[0][1], i[1][1]], [i[0][2], i[1][2]], color=c) )
  else:
    params = None
    callback = update_points_ghost
    geoms_real = ax.scatter(df_real[:,0,0], df_real[:,0,1], df_real[:,0,2], depthshade=False) # x,y,z vals
    geoms_pred = ax.scatter(df_pred[:,0,0], df_pred[:,0,1], df_pred[:,0,2], depthshade=False) # x,y,z vals
  return animation.FuncAnimation(fig,
    callback,
    frames,
    interval=speed,
    fargs=(geoms_real, df_real, geoms_pred, df_pred, params, ghost_shift),
    blit=False  
  ).to_html5_video()