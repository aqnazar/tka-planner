"""A scene, without a renderer.

What the Blender builder did to a ``bpy`` scene, this package does to a plain data
structure: place the bones, the cut planes, the axes, the landmarks, the implants and
the cutting blocks, then move them as the plan changes. Nothing here draws anything, so
the same scene serves a browser, a test, and later a tracker feed.
"""
