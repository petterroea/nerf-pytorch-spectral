from load_spectral_exr import load_exr, write_exr
import argparse

def main():
    parser = argparse.ArgumentParser(description='OpenEXR code test rig')
    parser.add_argument('infile', type=str, help='Input file')
    parser.add_argument('outfile', type=str, help='Output file')

    args = parser.parse_args()

    channels, skipchan, data = load_exr(args.infile)
    print("Loaded data %s" % str(data.shape))

    print("Channels %s, skipchan %s" % (channels, skipchan))

    write_exr(args.outfile, data, channels, skipchan)
    print("Saved")


if __name__ == '__main__':
    main()