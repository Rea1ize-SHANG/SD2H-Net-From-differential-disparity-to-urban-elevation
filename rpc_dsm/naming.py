def city_name(scene):
    scene = str(scene)
    if scene.startswith(('KM', 'GF7_')):
        return scene
    return 'GF7_' + scene
