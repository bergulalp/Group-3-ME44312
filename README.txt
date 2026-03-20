# NYC Yellow Taxi Analysis

This project processes and analyzes NYC Yellow Taxi data from January 2025. 

## Prerequisites

You must have Anaconda installed on your system to manage the environment and dependencies.

## Installation

1. Clone this repository to your local machine:
   git clone https://github.com/bergulalp/Group-3-ME44312.git
   cd Group-3-ME44312

2. Create the environment from the environment.yml file:

   Using Conda:
   conda env create -f taxi_environment.yml

   Using Mamba (recommended for speed):
   mamba env create -f taxi_environment.yml

3. Activate the environment:

   Using Conda:
   conda activate taxi_environment

   Using Mamba:
   mamba activate taxi_environment

## Development Environment

This project was developed using the Spyder IDE. However, you can use any IDE of your choice (such as VS Code or PyCharm) as long as you select the 'taxi_environment' as your interpreter.

## Data Requirements

The script expects the following file to be present in the project root directory:
- yellow_tripdata_2025-01.parquet

