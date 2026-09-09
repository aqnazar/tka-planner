"""Blender adapter: draw an engine scene, and nothing else.

What used to live here decided things. ``build.py`` placed components, cut bones, baked
an animation and managed a modifier stack. All of that now belongs to
:mod:`tka_planner.scene`, which needs no Blender at all, and the archived copy of the
old builder is kept at ``archive/blender-addon/`` as the port's reference.

What is left is a renderer and the import and export shims around it.
"""
