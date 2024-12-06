from colorama import Fore, init
from configparser import ConfigParser
from deid_shell import DeidShell

'''
DeID ToolKit main file
Contributors: Blaž Meden, Manfred Gonzalez, Esteban Leiva
'''

valid_hex = '0123456789ABCDEF'.__contains__

def cleanhex(data):
    return ''.join(filter(valid_hex, data.upper()))

def print_fore_fromhex(text, hexcode):
    """print in a hex defined color"""
    hexint = int(cleanhex(hexcode), 16)
    print("\x1B[38;2;{};{};{}m{}\x1B[0m".format(hexint>>16, hexint>>8&0xFF, hexint&0xFF, text))


if __name__ == '__main__':

    init()
    
    #print(Fore.GREEN + "DeID ToolKit")
    print("Starting " + Fore.WHITE + "De" + Fore.RED + "ID" + Fore.LIGHTGREEN_EX + " ToolKit ..." + Fore.RESET)
    #print(Fore.RED + "Contributors: Blaž Meden, Manfred Gonzalez, ...")
    
    # list all possible colors
    '''
    print(Fore.RED + "Red")
    print(Fore.GREEN + "Green")
    print(Fore.BLUE + "Blue")
    print(Fore.CYAN + "Cyan")
    print(Fore.MAGENTA + "Magenta")
    print(Fore.YELLOW + "Yellow")
    print(Fore.WHITE + "White")
    print(Fore.BLACK + "Black")
    print(Fore.LIGHTBLACK_EX + "Light Black")
    print(Fore.LIGHTBLUE_EX + "Light Blue")
    print(Fore.LIGHTCYAN_EX + "Light Cyan")
    print(Fore.LIGHTGREEN_EX + "Light Green")
    print(Fore.LIGHTMAGENTA_EX + "Light Magenta")
    print(Fore.LIGHTRED_EX + "Light Red")
    print(Fore.LIGHTWHITE_EX + "Light White")
    print(Fore.LIGHTYELLOW_EX + "Light Yellow")
    '''
    
    config = ConfigParser()
    config.read('config.ini')
    
    DeidShell(config=config).cmdloop()