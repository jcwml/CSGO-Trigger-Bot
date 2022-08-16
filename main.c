/*
--------------------------------------------------
    James William Fletcher (github.com/mrbid)
        AUGUST 2022

    Improved by Jim C. Williams. (github.com/jcwml)
--------------------------------------------------

    Trained using TBVGG3_ADA16 (github.com/TFNN)
    
    You can reduce ACTIVATION_SENITIVITY and increase REPEAT_ACTIVATION etc.

    Compile: clang aimbot.c -Ofast -mavx -mfma -lX11 -lm -o aim

*/

#include <unistd.h>
#include <stdint.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <sys/time.h>
#include <signal.h>

#include "TBVGG3_ADA16.h"

#pragma GCC diagnostic ignored "-Wgnu-folding-constant"

#define uint unsigned int
#define SCAN_AREA 28

const uint r0 = SCAN_AREA;  // dimensions of sample image square
const uint r2 = r0*r0;      // total pixels in square
const uint r2i = r2*3;      // total inputs to neural net pixels*channels
const uint rd2 = r0/2;      // total pixels divided by two
uint x=0, y=0;

float input[3][28][28];
    float r[r2] = {0};
    float g[r2] = {0};
    float b[r2] = {0};

Display *d;
int si;
Window twin;
GC gc = 0;
int tc = 0;
TBVGG3_Network net;

// hyperparameters that you can change
#define SCAN_VARIANCE 1
#define SCAN_DELAY 1000
#define ACTIVATION_SENITIVITY 0.55
#define REPEAT_ACTIVATION 0
#define FIRE_RATE_LIMIT_MS 600


/***************************************************
   ~~ Utils
*/
uint qRand(const uint min, const uint max)
{
    static float rndmax = 1.f/(float)RAND_MAX;
    return ( ( ((float)rand()) * rndmax ) * ((max+1)-min) ) + min;
}

uint64_t microtime()
{
    struct timeval tv;
    struct timezone tz;
    memset(&tz, 0, sizeof(struct timezone));
    gettimeofday(&tv, &tz);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

void rainbow_printf(const char* text)
{
    static unsigned int base_clr = 0;
    if(base_clr == 0)
        base_clr = (rand()%125)+55;
    
    base_clr += 3;

    unsigned int clr = base_clr;
    const unsigned int len = strlen(text);
    for(unsigned int i = 0; i < len; i++)
    {
        clr++;
        printf("\e[38;5;%im", clr);
        printf("%c", text[i]);
    }
    printf("\e[38;5;123m");
}

void rainbow_line_printf(const char* text)
{
    static unsigned int base_clr = 0;
    if(base_clr == 0)
        base_clr = (rand()%125)+55;
    
    printf("\e[38;5;%im", base_clr);
    base_clr++;
    if(base_clr >= 230)
        base_clr = (rand()%125)+55;

    const unsigned int len = strlen(text);
    for(unsigned int i = 0; i < len; i++)
        printf("%c", text[i]);
    printf("\e[38;5;123m");
}

//https://www.cl.cam.ac.uk/~mgk25/ucs/keysymdef.h
int key_is_pressed(KeySym ks)
{
    Display *dpy = XOpenDisplay(":0");
    char keys_return[32];
    XQueryKeymap(dpy, keys_return);
    KeyCode kc2 = XKeysymToKeycode(dpy, ks);
    int isPressed = !!(keys_return[kc2 >> 3] & (1 << (kc2 & 7)));
    XCloseDisplay(dpy);
    return isPressed;
}

void speakS(const char* text)
{
    char s[256];
    sprintf(s, "/usr/bin/espeak \"%s\"", text);
    if(system(s) <= 0)
        sleep(1);
}

Window findWindow(Display *d, Window current, char const *needle)
{
    Window ret = 0, root, parent, *children;
    unsigned cc;
    char *name = NULL;

    if(current == 0)
        current = XDefaultRootWindow(d);

    if(XFetchName(d, current, &name) > 0)
    {
        if(strstr(name, needle) != NULL)
        {
            XFree(name);
            return current;
        }
        XFree(name);
    }

    if(XQueryTree(d, current, &root, &parent, &children, &cc) != 0)
    {
        for(unsigned int i = 0; i < cc; ++i)
        {
            Window win = findWindow(d, children[i], needle);

            if(win != 0)
            {
                ret = win;
                break;
            }
        }
        XFree(children);
    }
    return ret;
}

void processScanArea(Window w)
{
    // get image block
    XImage *img;
#ifdef SCAN_VARIANCE
    img = XGetImage(d, w, (x+qRand(-SCAN_VARIANCE, SCAN_VARIANCE))-rd2, (y+qRand(-SCAN_VARIANCE, SCAN_VARIANCE))-rd2, r0, r0, AllPlanes, XYPixmap);
#else
    img = XGetImage(d, w, x-rd2, y-rd2, r0, r0, AllPlanes, XYPixmap);
#endif
    if(img == NULL)
        return;

    // colour map
    const Colormap map = XDefaultColormap(d, si);

    // extract colour information
    int i = 0;
    for(int y = 0; y < r0; y++)
    {
        for(int x = 0; x < r0; x++)
        {
            XColor c;
            c.pixel = XGetPixel(img, x, y);
            XQueryColor(d, map, &c);

            r[i] = (float)c.red;
            g[i] = (float)c.green;
            b[i] = (float)c.blue;

            i++;
        }
    }

    // free image block
    XFree(img);

    /////////////////
    // -1 to 1 normalised
    for(uint j = 0; j < 28; j++)
    {
        for(uint k = 0; k < 28; k++)
        {
            input[0][j][k] = (r[(j*28)+k]-128.f)*0.003921568859f;
            input[1][j][k] = (g[(j*28)+k]-128.f)*0.003921568859f;
            input[2][j][k] = (b[(j*28)+k]-128.f)*0.003921568859f;
        }
    }
}

/***************************************************
   ~~ Program Entry Point
*/
int main(int argc, char *argv[])
{
    srand(time(0));
    signal(SIGPIPE, SIG_IGN);

    // intro
    rainbow_printf("James William Fletcher (github.com/mrbid)\n");
    rainbow_printf("Jim C. Williams (github.com/jcwml)\n\n");
    rainbow_printf("L-CTRL + L-ALT = Toggle BOT ON/OFF\n");
    rainbow_printf("R-CTRL + R-ALT = Toggle HOTKEYS ON/OFF\n");
    rainbow_printf("P = Toggle crosshair.\n");
    rainbow_printf("G = Get activation for reticule area.\n");
    rainbow_printf("H = Get scans per second.\n");
    printf("\e[38;5;76m");
    printf("\nMake the crosshair a single green pixel.\nOR disable the game crosshair and use the crosshair provided by this bot.\nOR if your monitor provides a crosshair use that. (this is best)\n\n");
    printf("This bot will only auto trigger when W,A,S,D & L-SHIFT are not being pressed.\n(so when your not moving in game, aka stationary)\n\nL-SHIFT allows you to disable the bot while stationary if desired.\n\n");
    printf("This dataset is trained only on Counter-Terrorist heads\nso you need to play as Terrorist forces.\n\n");
    printf("\e[38;5;123m");

    // open display 0
    d = XOpenDisplay(":0");
    if(d == NULL)
    {
        printf("Failed to open display\n");
        return 0;
    }

    // get default screen
    si = XDefaultScreen(d);

    // get graphics context
    gc = DefaultGC(d, si);

    // find window
    twin = findWindow(d, 0, "Counter-Strike");
    if(twin != 0)
        printf("CS:GO Win: 0x%lX\n\n", twin);

    // load network
    if(TBVGG3_LoadNetwork(&net, "a83.save") < 0)
        printf("!! Starting with no training data. (failed to load network file)\n\n");

    //
    
    XEvent event;
    memset(&event, 0x00, sizeof(event));
    
    uint enable = 0;
    uint crosshair = 0;
    uint hotkeys = 1;

    //
    
    while(1)
    {
        // loop every SCAN_DELAY ms (1,000 microsecond = 1 millisecond)
        usleep(SCAN_DELAY);

        // bot toggle
        if(key_is_pressed(XK_Control_L) && key_is_pressed(XK_Alt_L))
        {
            if(enable == 0)
            {
                 // get window
                twin = findWindow(d, 0, "Counter-Strike");

                // get center window point (x & y)
                XWindowAttributes attr;
                XGetWindowAttributes(d, twin, &attr);
                x = attr.width/2;
                y = attr.height/2;

                // set mouse event
                memset(&event, 0x00, sizeof(event));
                event.type = ButtonPress;
                event.xbutton.button = Button1;
                event.xbutton.same_screen = True;
                event.xbutton.subwindow = twin;
                event.xbutton.window = twin;

                enable = 1;
                usleep(300000);
                rainbow_line_printf("BOT: ON\n");
                speakS("on");
            }
            else
            {
                enable = 0;
                usleep(300000);
                rainbow_line_printf("BOT: OFF\n");
                speakS("off");
            }
        }
        
        // toggle bot on/off
        if(enable == 1)
        {
            // print samples per second when pressed
            if(key_is_pressed(XK_H))
            {
                static uint64_t st = 0;
                static uint sc = 0;
                processScanArea(twin);
                const float ret = TBVGG3_Process(&net, &input[0], NO_LEARN);
                sc++;
                if(microtime() - st >= 1000000)
                {
                    printf("\e[36mSPS: %u\e[0m\n", sc);
                    sc = 0;
                    st = microtime();
                }
                continue;
            }

            // input toggle
            if(key_is_pressed(XK_Control_R) && key_is_pressed(XK_Alt_R))
            {
                if(hotkeys == 0)
                {
                    hotkeys = 1;
                    usleep(300000);
                    printf("HOTKEYS: ON [%ix%i]\n", x, y);
                    speakS("hk on");
                }
                else
                {
                    hotkeys = 0;
                    usleep(300000);
                    rainbow_line_printf("HOTKEYS: OFF\n");
                    speakS("hk off");
                }
            }

            if(hotkeys == 1)
            {
                // crosshair toggle
                if(key_is_pressed(XK_P))
                {
                    if(crosshair == 0)
                    {
                        crosshair = 1;
                        usleep(300000);
                        rainbow_line_printf("CROSSHAIR: ON\n");
                        speakS("cx on");
                    }
                    else
                    {
                        crosshair = 0;
                        usleep(300000);
                        rainbow_line_printf("CROSSHAIR: OFF\n");
                        speakS("cx off");
                    }
                }
            }
            
            if(hotkeys == 1 && key_is_pressed(XK_G)) // print activation when pressed
            {
                processScanArea(twin);
                const float ret = TBVGG3_Process(&net, &input[0], NO_LEARN)*0.01f;
                if(ret > ACTIVATION_SENITIVITY)
                {
                    printf("\e[93mA: %f\e[0m\n", ret);
                    XSetForeground(d, gc, 65280);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                    XSetForeground(d, gc, 0);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                    XFlush(d);
                }
                else
                {
                    printf("\e[0mA: %f\n", ret);
                    XSetForeground(d, gc, 16711680);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                    XSetForeground(d, gc, 0);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                    XFlush(d);
                }
            }
            else
            {
                if(crosshair == 1)
                {
                    XSetForeground(d, gc, 65280);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                    XSetForeground(d, gc, 0);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                    XFlush(d);
                }

                if(key_is_pressed(XK_W) == 0 && key_is_pressed(XK_A) == 0 && key_is_pressed(XK_S) == 0 && key_is_pressed(XK_D) == 0 && key_is_pressed(XK_Shift_L) == 0)
                {
                    processScanArea(twin);
                    int TRIGGER = 1; // low detections get one shot
                    const float activation = TBVGG3_Process(&net, &input[0], NO_LEARN)*0.01f;
                    // if(activation > 0.8){TRIGGER = 3;} // mid range gets 3 shots
                    // if(activation > 0.9){TRIGGER = 6;} // high range gets 6 shots

                    // passed minimum activation?
                    if(activation > ACTIVATION_SENITIVITY)
                    {
                        tc++;

                        // did we activate enough times in a row to be sure this is a target?
                        if(tc > REPEAT_ACTIVATION)
                        {
                            // fire off as many shots as we need to
                            for(int i = 0; i < TRIGGER; i++)
                            {
                                // fire mouse down
                                event.type = ButtonPress;
                                event.xbutton.state = 0;
                                XSendEvent(d, PointerWindow, True, 0xfff, &event);
                                XFlush(d);
                                
                                // wait 100ms (or ban for suspected cheating)
                                usleep(100000);
                                
                                // release mouse down
                                event.type = ButtonRelease;
                                event.xbutton.state = 0x100;
                                XSendEvent(d, PointerWindow, True, 0xfff, &event);
                                XFlush(d);

                                // fire limit
                                usleep(FIRE_RATE_LIMIT_MS * 100);
                            }

                            // fire limit
                            usleep(FIRE_RATE_LIMIT_MS * 1000);
                        }
                    }
                    else
                    {
                        tc = 0;
                    }
                }
            }

        ///
        }
    }

    // done, never gets here in regular execution flow
    return 0;
}
