import stereo_rectification
import semantic_segmentation
import location_torch
import reconstruction
import model_icp

def main():
    group = 6
    stereo_rectification.stereo_rect(group)
    semantic_segmentation.seg()
    location_torch.output_model(group)
    reconstruction.main()
    x,y,z,w,t = model_icp.main()
    mess = '0{q0[%f] q1[%f] q2[%f] q3[%f] x[%f] y[%f] z[%f]}'%(x,y,z,w,t[0],t[1],t[2])
    # # mess = '0{q0[0] q1[0] q2[0] q3[0] x[0] y[0] z[0]}'
    return mess

if __name__ == '__main__':
    mess = main()
