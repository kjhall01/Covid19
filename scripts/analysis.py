import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature, LAND, COASTLINE
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy import feature
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import os
from pygifsicle import optimize
import sys
from skimage import transform,io
import matplotlib.image as mpimg
import imageio
import matplotlib.colors as colors
from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings



warnings.filterwarnings("ignore")


debug_count = 1
def debug():
    global debug_count
    print("Profile Point {}".format(debug_count))
    debug_count += 1


def get_date(data, day, datatype):
    datatypes = {'total cases':3, "total deaths":4, "new cases":5, "new deaths":6}
    for i in range(len(data)):
        if str(data[i][0]) == day:
            return data[i][datatypes[datatype]]
    return 0


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def get_county(county_name, state_name, today):
    try:
        unj = np.genfromtxt('../county_data/{}_{}_{}.csv'.format(county_name, state_name, today), skip_header=1, delimiter=',')
        if unj.shape[0] == 0:
            raise FileNotFoundError
        union_nj = [[date(int(unj[i,0].split('-')[0]),int(unj[i,0].split('-')[1]), int(unj[i,0].split('-')[2])), unj[i,1], unj[i,2], int(unj[i,4]), int(unj[i,5])] for i in range(unj.shape[0])]
        union_nj = sorted(union_nj, key=lambda x: x[0])
        for i in range(len(union_nj)):
            if i == 0:
                union_nj[i].extend([0,0])
            else:
                union_nj[i].extend([(union_nj[i][3] - union_nj[i-1][3]), (union_nj[i][4] - union_nj[i-1][4])])
        return union_nj
    except:
        df1 = pd.read_csv('../covid-19-data/us-counties.csv')
        county = df1.to_numpy(dtype=str)
        union = county[county[:,1] == county_name]
        unj = union[union[:,2] == state_name]
        union_nj = [[date(int(unj[i,0].split('-')[0]),int(unj[i,0].split('-')[1]), int(unj[i,0].split('-')[2])), unj[i,1], unj[i,2], int(unj[i,4]), int(unj[i,5])] for i in range(unj.shape[0])]
        union_nj = sorted(union_nj, key=lambda x: x[0])
        for i in range(len(union_nj)):
            if i == 0:
                union_nj[i].extend([0,0])
            else:
                union_nj[i].extend([(union_nj[i][3] - union_nj[i-1][3]), (union_nj[i][4] - union_nj[i-1][4])])
        union_nj.insert(0, ['date', 'county', 'state', 'confirmed cases', 'deaths', 'new cases per day', 'new deaths per day'])
        union_nj2 = np.asarray(union_nj, dtype=str)
        np.savetxt('../county_data/{}_{}_{}.csv'.format(county_name, state_name, today), union_nj2, delimiter=',', fmt='%s')
        return union_nj

def main(today):
    #set parameters
    #nla, sla, wlo, elo = 45, 37, -79, -71 #broad view
    nla, sla, wlo, elo = 49.7, 23.3, -125, -64.3 #USA
    x_dim, y_dim = 50, 50
    states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming' }
    #prepare to "geocode"
    df = pd.read_csv('../shape raw/geocodes.csv')
    df = df[df['longitude'] > wlo-1]
    df = df[df['longitude'] < elo+1]
    df = df[df['latitude'] > sla-1]
    df = df[df['latitude'] < nla+1]
    df = df.sort_values(['latitude', 'longitude'])
    df_n = df.to_numpy()
    x_s, y_s = np.linspace(wlo, elo, num=x_dim), np.linspace(nla, sla, num=y_dim)
    base_grid_x, base_grid_y = np.meshgrid(x_s, y_s)
    base_grid = np.dstack((base_grid_y, base_grid_x))

    #geocode
    county_grid = np.zeros(base_grid.shape[0] * base_grid.shape[1]).reshape((x_dim, y_dim))
    county_grid = np.asarray(county_grid, dtype=str)
    print('[', end='')
    for i in range(base_grid.shape[0]):
        for j in range(base_grid.shape[1]):
            cur_dist = 100000000
            for k in range(df_n.shape[0]):
                dist = np.sqrt((base_grid[i,j,0] - df_n[k,3])**2 + (base_grid[i,j,1] - df_n[k,4])**2)
                if dist <= cur_dist:
                    cur_dist = dist
                    county_grid[i,j] = '{},{}'.format(df_n[k,2], df_n[k,5])
            if ((i*base_grid.shape[0] + j) / float(base_grid.shape[0]*base_grid.shape[1]) * 100) % 10 == 0:
                print('=', end='')
                sys.stdout.flush()
    print(']')
    sys.stdout.flush()

    days = np.genfromtxt('../county_data/dates.csv', delimiter=',', skip_header=0, dtype=str)
    days = days.reshape(-1,1)
    it = 1
    while days[it,0] != today:
        it += 1
    titles = ['Confirmed Cases', 'Daily New Cases', 'Deaths', 'Daily New Deaths']
    for g in range(it):
        dayo = days[g,0]
        print(dayo)
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,20), subplot_kw={'projection': ccrs.PlateCarree()})
        for yi in range(2):
            for xi in range(2):
                grid = [[0 for i in range(county_grid.shape[0])] for j in range(county_grid.shape[1])]
                for i in range(county_grid.shape[0]):
                    for j in range(county_grid.shape[1]):
                        state, county = county_grid[i,j].split(',')
                        state = states[state]
                        data = get_county(county, state, dayo)
                        datatypes = {3:'total cases', 4:"total deaths", 5:"new cases", 6:"new deaths"}
                        grid[i][j] = get_date(data, dayo, datatypes[xi*2+yi+3 ])
                np.savetxt('../output_data/{}_{}_USA.csv'.format(dayo, titles[xi*2+yi]), grid,delimiter=',', fmt='%s')

                #from https://stackoverflow.com/questions/51106763/county-boarders-in-cartopy
                #reader = shpreader.Reader('../shapefiles/countyl010g.shp')
                #counties = list(reader.geometries())
                #COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
                ax[yi][xi].set_title(titles[xi*2+yi])
                ax[yi][xi].set_extent([wlo,elo,nla,sla], ccrs.PlateCarree())
                states_provinces = feature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_shp', scale='10m', facecolor='none')
                ax[yi][xi].add_feature(feature.LAND)
                #ax[yi][xi].add_feature(COUNTIES, facecolor='none', edgecolor='gray',zorder=120)
                pl=ax[yi][xi].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                pl.xlabels_top, pl.ylabels_left, pl.ylabels_right, pl.xlabels_bottom = False, False, False, True
                pl.xformatter, pl.yformatter  = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
                ax[yi][xi].add_feature(states_provinces, edgecolor='black', linewidth=0.75, zorder=130)
                ax[yi][xi].set_ybound(lower=sla, upper=nla)
                if yi % 2 == 0:
                    top = max(max(grid))
                    current_cmap = plt.get_cmap('Spectral_r',28)
                    vmini=0
                    if top < 50:
                        vmaxi = 50
                    elif top < 100:
                        vmaxi=100
                    elif top < 250:
                        vmaxi = 250
                    elif top < 500:
                        vmaxi = 500
                    elif top < 1000:
                        vmaxi = 1000
                    else:
                        vmaxi = 10000
                    vmaxi = 1000
                else:
                    top = max(max(grid))
                    current_cmap = plt.get_cmap('Spectral_r',28)
                    vmini=0
                    if top < 5:
                        vmaxi = 5
                    elif top < 10:
                        vmaxi=10
                    elif top < 25:
                        vmaxi = 25
                    elif top < 50:
                        vmaxi = 50
                    elif top < 100:
                        vmaxi = 100
                    else:
                        vmaxi = 500
                    vmaxi = 50
                if top > 0:
                    CS = ax[yi][xi].pcolormesh(base_grid_x, base_grid_y, grid,
                        vmin=vmini, vmax=vmaxi,
                        norm=MidpointNormalize(midpoint=0.),
                        cmap=current_cmap)
                else:
                    CS = ax[yi][xi].pcolormesh(base_grid_x, base_grid_y, grid,
                        vmin=vmini, vmax=vmaxi,
                        norm=MidpointNormalize(midpoint=0.),
                        cmap=current_cmap)
                ax[yi][xi].add_feature(feature.OCEAN, zorder=100)
                axins_det = inset_axes(ax[yi][xi],
                    width="5%",  # width = 5% of parent_bbox width
                    height="100%",  # height : 50%
                    loc='center left',
                    bbox_to_anchor=(-0.15, 0., 1, 1),
                    bbox_transform=ax[yi][xi].transAxes,
                    borderpad=0.1 )
                #top = max(max(grid))
                #if top < 14:
                #    top = 14
                cbar_ldet = fig.colorbar(CS, ax=ax[yi][xi], cax=axins_det,  orientation='vertical', pad=0.02)
                cbar_ldet.set_label('Lives')
                axins_det.yaxis.tick_left()
        fig.subplots_adjust(wspace=0.3, hspace=0.1)
        fig.suptitle(dayo, fontsize=48)
        plt.savefig('../output/{}_USA.png'.format(dayo), dpi=300)
    print('done')


def graph(today, state, counties):
    data = []
    for county in counties:
        data.append( get_county( county, state, today))
    first_date = today
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][0] != 'date':
                if data[i][j][0] < first_date:
                    first_date = data[i][j][0]
    days = int((today - first_date).days)
    dates = [ first_date + timedelta(days=i) for i in range(days + 1) ]

    data2 = []
    for i in range(len(data)):
        data2.append([])
        for j in range(len(data[i])):
            if data[i][j][0] != 'date':
                data2[i].append([dates.index(data[i][j][0]), data[i][j][3], data[i][j][4], data[i][j][5], data[i][j][6]])
        #data2[i] = np.asarray(data2[i])

    colors = ['b','g','r','c','m','y','k']
    linestyles = ['-','--','-.',':']
    mystyles = []
    for i in range(len(linestyles)):
        mystyles.extend([linestyles[i]+colors[j] for j in range(len(colors))])
    d2_filled = data2

    days2_int = [i for i in range(days+1)]
    for i in range(len(d2_filled)):
        missing = []
        j = 0
        while j != days2_int[-1]:
            if d2_filled[i][j][0] != days2_int[j]:
                d2_filled[i].insert(j, [dates[j], 0,0,0,0])
            else:
                d2_filled[i][j][0] = dates[j]
            j += 1
        d2_filled[i][j][0] = dates[j]
        d2_filled[i] = np.asarray(d2_filled[i])
    d2_filled = np.asarray(d2_filled)
    d2m = np.mean(d2_filled[:,:,1:], axis=0)
    d2m = np.hstack((np.asarray(dates).reshape(-1,1), d2m))
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
    fig.suptitle(state)
    titles = ['Confirmed Cases', 'Deaths', "Daily New Cases", 'Daily New Deaths']
    for i in range(2):
        for j in range(2):
            ax[i][j].set_title(titles[i*2+j])
            for k in range(len(counties)):
                ax[i][j].plot(d2_filled[k,:,0], d2_filled[k,:,i*2+j+1], mystyles[k], label=counties[k])
            ax[i][j].plot(d2m[:,0], d2m[:,i*2+j+1], mystyles[k+1], label='State Mean')
            ax[i][j].legend(loc=2)
            plt.xticks(rotation=70)
    #fig.autofmt_xdate()
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=35)
    plt.savefig('../output/{}_graphs.png'.format(state), dpi=300)


def display(today):
    #today = string YYYY-MM-DD
    days = np.genfromtxt('../county_data/dates.csv', delimiter=',', skip_header=0, dtype=str)
    days = days.reshape(-1,1)
    j = 1
    while days[j,0] != today:
        j += 1
    imgs = []
    for i in range(j):
        day = days[i,0]
        img = mpimg.imread('../output/{}_fixed.png'.format(day))
        imgs.append(img)
    for img in imgs:
        imgplot = plt.imshow(img)
        plt.show(block=False)
        plt.pause(0.16)
        plt.close()

def downscale_images(filenames):
    images = []
    for filename in filenames:
        print(filename)
        im = io.imread(filename)
        #https://stackoverflow.com/questions/44257947/skimage-weird-results-of-resize-function
        im = transform.rescale(im, 0.25, multichannel=True)
        im = 255 * im
        im = im.astype(np.uint8)
        io.imsave('../output/{}_USA_downscaled.png'.format(filename), im)
        im = imageio.imread('../output/{}_USA_downscaled.png'.format(filename))
        images.append(im)
    return images


def save_gif(today):
    #today = string YYYY-MM-DD
    days = np.genfromtxt('../county_data/dates.csv', delimiter=',', skip_header=0, dtype=str)
    days = days.reshape(-1,1)
    i = 0
    while days[i,0] != today:
        i += 1
    filenames = ['../output/{}_USA.png'.format(days[j, 0]) for j in range(i)  ]
    images = downscale_images(filenames)
    imageio.mimsave('../output/covid_spread_usa.gif', images)
    optimize('../output/covid_spread_usa.gif')


if __name__ == "__main__":
    counties = ['Union', 'Morris', 'Essex', 'Bergen', 'Hunterdon', 'Somerset', 'Monmouth', 'Middlesex', 'Gloucester', 'Atlantic', 'Hudson', 'Ocean','Passaic', 'Burlington', 'Mercer', 'Camden', 'Cape May', 'Cumberland', 'Warren', 'Salem']
    today = date(2020, 4, 3)
    graph(today, 'New Jersey', counties)
    #save_gif('2020-04-03')
    #main()
    #main('2020-04-03')
