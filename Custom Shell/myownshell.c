/** \file myownshell.c 
	* Project OS 2015
	*Papavasileiou Themistoklis (7375)
	*thpapavasileiou@gmail.com
	*/



#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAXARGS 6 ///< maximum number of commandline arguments
#define SIZE 100 // number of allowed background procesess

pid_t global_pid;  // Global is frowned upon, but signal handlers are really low level, and it's a good tradeoff not to mess with them.
/** \brief Parser of the arguments.
	
	*Scans the whole NULL terminated input, makes whitespace equal to zero.
	*When it finds something that's not whitespace, it keeps track of its location,since it must be a word.
	*It also keeps track of how many times it does that, which servers as an argument counter.
	*Then, it loops over the whole word (till it finds a whitespace character) and sets the line pointer accordingly.
	*Finally, it makes the args array NULL terminated and returns the argument counter.

*/
int parse( char * line, char ** args)

{
    int argc=0; 
    while( *line != '\0' ) 
    {
        while( *line == ' ' || *line == '\t' || *line == '\n') *line++ = '\0';
        *args++ = line;argc++; 
        
        while( *line != '\0' && *line != ' ' && *line != '\t' && *line != '\n')  line++;
    }
    *args = NULL;   
    return argc;
}
/** @brief Handler of the SIGINT signal
	*specifies what to do, when a signal we define later is 'caught'
	*In this case we just kill it.
*/

void handler( )
{
    
    kill(SIGTERM,global_pid);
    
}
/** @brief Execution part of the shell (Foreground)

	*This is where the parsed arguments come to be executed.
	*After we fork and check for errors, we monitor the child process, and wait for it to finish.
	*In order to be able to interrupt the process with a Ctrl-C action, we must check before the waiting takes place, with
	*a signal function. This time we're doing POSIX, so it's okay to use signal(). A 'better' way would be sigaction, since it
	*behaves better across platforms. For now this will do nicely!
*/

void execute( char **args) 
{
    printf(" // executing in the foreground //\n");
    pid_t pid;
    int status;
    pid=fork();
    global_pid=pid;
    if( pid<0){ printf("forking error!\n"); }
    else if( pid == 0 )
    {
        if( execvp(args[0],args) < 0)
        {
            printf("execution error!\n");
            exit(1);
        }
    }
    else
    {
        //fork returns pid instead of 0, since we are in the parent process now
        //so we wait and do nothing, till our children terminate, in a normal  way, or not.
        signal(SIGINT,handler);
        while( wait(&status) != pid ){;}
    }

}
/** @brief Execution, in the background

	*same principle as foreground execution, only now we do not wait for the forked process to finish,
	*PID of mother is also printed, as a form of feedback for the user.
*/

int backgroundExecute(char **args,int argc)
{
    args[argc-1]=NULL;
    pid_t pid;
    pid=fork();
    if( pid<0 ){ printf("forking error!!\n"); }
    else if( pid==0 ) 
    {
        if( execvp(args[0],args) <0 )
        {
            printf("background execution error!\n");
            exit(2);
        }
    }
    else{ 
    	printf(" PID IS %d \n",pid);
		return pid;
		} 
}
/** @brief Main function

	*First we initialize a command line string (empty), an argument array, an argument counter and a string to hold the current working directory.
	*Then an infinite loop checks for the Ctrl-C action as requested, and we are presented with the cwd and a prompt.
	*The input string is zero-terminated then passed on to the parser for processing. After that, we check for empty strings
	*since not skipping the iteration if the string is empty gives segmentation faults. Two more checks take place, for the cd and exit commands
	*Finally, we examine the last argument of our parsed input and decide whether it is to be executed on the foreground or not.
	*Note that char line[1000] hols the input before the parser, and after that it's handled by the *args[] array.

	*The abstract logic behind the main function is, get input -> parse it -> execute it -> repeat.
*/

//POSIX SIGCHLD handler
void handle_sigchld(int sig) {
	printf("in the handler, where background processes go to die \n");
  while (waitpid((pid_t)(-1), 0, WNOHANG) > 0) {}
  	main();//return us to prompt!
}

// takes all pids and sends a termination signal with visual feedback!
void cleanup(pid_t *children)
{
	int i;
	for(i=0;i<SIZE;i++)
	{
		if(!kill(children[i],SIGTERM)){printf("%d terminated\n",children[i]);}
	}
}

int main(void)
{
	int counter =0;
	pid_t *children = (pid_t *)malloc(SIZE*sizeof(pid_t));
	memset(children,0,SIZE);
	// POSIX for sigchld handling
	struct sigaction sa;
	sa.sa_handler = &handle_sigchld;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
	if (sigaction(SIGCHLD, &sa, 0) == -1) {
  		perror(0);
  		exit(1);
	}
    char line[1000]="";
    char *args[MAXARGS];
    int argc=0;
    char cwd[1000];

    while(1)
    {
    // check for background procs and Ctrl-C.
    signal(SIGINT,handler);
    sigaction(SIGCHLD, &sa, 0);
    getcwd(cwd,sizeof(cwd)); 
    printf("[ %s ] ",cwd);
    printf("Shelly > ");
    fgets(line,sizeof(line),stdin);
    printf("\n");
    //remove newline imposed by fgets. Super important!
    line[ strlen(line)-1 ] = '\0';
    argc=parse(line,args);
    if ( argc==0 ){ continue; }
    if ( strcmp(args[0] , "exit") == 0 ){ sigaction(SIGCHLD, &sa, 0);cleanup(children);exit(0);} //make sure we kill everything before we leave!
    if ( strcmp(args[0] , "cd") == 0 ){ chdir(args[1]);continue; }
    if ( strcmp(args[argc-1], "&") == 0 ){ children[counter++]=backgroundExecute(args,argc); }
    else{ execute(args); }
    }
}

