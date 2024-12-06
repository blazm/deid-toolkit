from ErrorHandler import DeidtoolkitError

try:
    raise DeidtoolkitError("Esto es una prueba", "preprocesing")
     
except DeidtoolkitError as e: 
    print(e)