import logging

global logging

def config_log(s):
	logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
	    filename= s, 
	    level=logging.INFO)



def do_something():
    logging.info('Doing something')

def main():
    s = 'tt.txt'
    config_log(s)
    logging.info('Started')
    do_something()
    logging.info('Finished')

if __name__ == '__main__':
    main()
