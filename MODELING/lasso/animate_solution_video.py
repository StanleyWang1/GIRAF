# animate_solution.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import matplotlib.transforms as mtransforms
from pathlib import Path
import matplotlib.animation as animation  # Add this import

from params import R_t, dt  # R_t doubles as the visualization radius/size

# ---- configuration knobs ----
solution_file = "./MODELING/lasso/solutions/soln_down.npz"
omega_target = 2.0
show_vehicle_heading = False
trail_len = 25        # windowed trails for target/base; tip trail = full

# ---- helpers ----
def boom_tip(x_v, y_v, th_v, th1, d2):
    phi = th_v + th1
    return x_v + d2*np.cos(phi), y_v + d2*np.sin(phi)

def extract_series(X_opt):
    x_t, y_t = X_opt[0,:], X_opt[1,:]
    x_v, y_v = X_opt[2,:], X_opt[3,:]
    th_v, th1, d2 = X_opt[4,:], X_opt[5,:], X_opt[6,:]
    tip_x, tip_y = boom_tip(x_v, y_v, th_v, th1, d2)
    return x_t, y_t, x_v, y_v, th_v, th1, d2, tip_x, tip_y

def main():
    p = Path(solution_file)
    if not p.exists():
        raise FileNotFoundError(f"Could not find {solution_file}")
    
    # Get output filename from solution path
    mp4_path = p.with_suffix('.mp4')
    data = np.load(p, allow_pickle=True)
    X_opt = data["X_opt"]
    x_t, y_t, x_v, y_v, th_v, th1, d2, tip_x, tip_y = extract_series(X_opt)
    T = X_opt.shape[1]
    t_arr = np.arange(T)*dt

    # Add 0.5 seconds of final pose
    extra_frames = int(0.5 / dt)  # Number of frames for 0.5 seconds
    x_t = np.append(x_t, [x_t[-1]] * extra_frames)
    y_t = np.append(y_t, [y_t[-1]] * extra_frames)
    x_v = np.append(x_v, [x_v[-1]] * extra_frames)
    y_v = np.append(y_v, [y_v[-1]] * extra_frames)
    th_v = np.append(th_v, [th_v[-1]] * extra_frames)
    th1 = np.append(th1, [th1[-1]] * extra_frames)
    d2 = np.append(d2, [d2[-1]] * extra_frames)
    tip_x = np.append(tip_x, [tip_x[-1]] * extra_frames)
    tip_y = np.append(tip_y, [tip_y[-1]] * extra_frames)
    T = len(x_t)  # Update total frames
    t_arr = np.arange(T)*dt

    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect("equal")

    # Target
    target_disc = Circle((0,0), R_t, facecolor="lightskyblue",
                         edgecolor="royalblue", lw=2, alpha=0.5)
    ax.add_patch(target_disc)
    target_spin_line, = ax.plot([],[],"-",lw=1.5,color="royalblue")

    # Vehicle
    veh_side = 1.5*R_t
    veh_square = Rectangle((-veh_side/2,-veh_side/2),veh_side,veh_side,
                            facecolor="seagreen",edgecolor="seagreen",
                            lw=2,alpha=0.5)
    ax.add_patch(veh_square)

    # Boom + tip
    boom_line, = ax.plot([],[],"-",lw=3.75,color="gray")
    tip_dot,  = ax.plot([],[],"o",ms=9,color="darkred")

    # Trails
    if trail_len>0:
        trail_target, = ax.plot([],[],":",lw=1.4,alpha=0.9,color="royalblue")  # Changed style to match boom
        trail_base,   = ax.plot([],[],":",lw=1.4,alpha=0.9,color="seagreen")   # Changed style to match boom
    # full dotted boom-endpoint trail
    trail_tip_full, = ax.plot([],[],":",lw=1.4,alpha=0.9,color="darkred")

    xs=np.concatenate([x_t,x_v,tip_x]); ys=np.concatenate([y_t,y_v,tip_y])
    pad=0.5+0.1*max(1.0,np.nanmax(np.hypot(xs-xs.mean(),ys-ys.mean())))
    ax.set_xlim(xs.min()-pad,xs.max()+pad); ax.set_ylim(ys.min()-pad,ys.max()+pad)
    ax.grid(True,alpha=0.3)
    ax.set_title("Acquisition with Boom — Optimal Approach")
    time_text=ax.text(0.02,0.98,"",transform=ax.transAxes,va="top",ha="left")

    def init():
        target_disc.center=(np.nan,np.nan)
        target_spin_line.set_data([],[])
        veh_square.set_transform(ax.transData)
        boom_line.set_data([],[]); tip_dot.set_data([],[])
        if trail_len>0:
            trail_target.set_data([],[]); trail_base.set_data([],[])
        trail_tip_full.set_data([],[])
        time_text.set_text("")
        return target_disc,target_spin_line,veh_square,boom_line,tip_dot,trail_tip_full,time_text

    def update(k):
        cx,cy=x_t[k],y_t[k]; target_disc.center=(cx,cy)
        theta=omega_target*t_arr[k]
        target_spin_line.set_data([cx,cx+R_t*np.cos(theta)],[cy,cy+R_t*np.sin(theta)])

        vx,vy,th=x_v[k],y_v[k],th_v[k]
        veh_square.set_transform(mtransforms.Affine2D().rotate(th).translate(vx,vy)+ax.transData)

        boom_line.set_data([vx,tip_x[k]],[vy,tip_y[k]])
        tip_dot.set_data([tip_x[k]],[tip_y[k]])

        # persistent dotted trail for tip
        trail_tip_full.set_data(tip_x[:k+1],tip_y[:k+1])

        if trail_len>0:
            i0=max(0,k-trail_len)
            trail_target.set_data(x_t[i0:k+1],y_t[i0:k+1])
            trail_base.set_data(x_v[i0:k+1],y_v[i0:k+1])

        time_text.set_text(f"t = {k*dt:.2f} s")
        return target_disc,target_spin_line,veh_square,boom_line,tip_dot,trail_tip_full,time_text

    ani = FuncAnimation(fig, update, frames=T, init_func=init,
                       blit=False, interval=1000*dt, repeat=True, repeat_delay=1000)
    
    # Save animation first
    print(f"[info] Saving animation to {mp4_path}")
    writer = animation.PillowWriter(fps=10)
    ani.save(str(mp4_path).replace('.mp4', '.gif'), writer=writer)  # Save as GIF instead
    print("[done] Animation saved successfully")
    
    # Then show interactive window
    plt.show()

if __name__=="__main__":
    main()
