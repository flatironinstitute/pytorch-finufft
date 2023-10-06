pipeline {
  agent none
  options {
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
    timeout(time: 1, unit: 'HOURS')
  }
  stages {
    stage('CUDA Tests') {
      agent {
         dockerfile {
            filename 'ci/docker/Dockerfile-cuda11.8'
            args '--gpus 2'
            label 'docker && v100'
         }
      }
      environment {
    HOME = "$WORKSPACE"
    PYBIN = "/opt/python/cp38-cp38/bin"
    LIBRARY_PATH = "$WORKSPACE/finufft/build"
    LD_LIBRARY_PATH = "$WORKSPACE/finufft/build"
      }
      steps {

    // TODO - reconsider install strategy once finufft/cufinufft 2.2 is released
  checkout scmGit(branches: [[name: '*/master']],
                  extensions: [cloneOption(noTags: true, reference: '', shallow: true),
                               [$class: 'RelativeTargetDirectory', relativeTargetDir: 'finufft'],
                               cleanAfterCheckout()],
                  userRemoteConfigs: [[url: 'https://github.com/flatironinstitute/finufft']])

    sh '''#!/bin/bash -ex
      nvidia-smi
    '''
    sh '''#!/bin/bash -ex
      echo $HOME
      ls
    '''
    sh '''#!/bin/bash -ex
        cd finufft
        # v100 cuda arch
        cuda_arch="70"

        cmake -B build . -DFINUFFT_USE_CUDA=ON \
                         -DFINUFFT_USE_CPU=OFF \
                         -DFINUFFT_BUILD_TESTS=ON \
                         -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
                         -DBUILD_TESTING=ON
        cd build
        make -j4
    '''

    sh '${PYBIN}/python3 -m venv $HOME'
    sh '''#!/bin/bash -ex
      source $HOME/bin/activate
      python3 -m pip install --upgrade pip
      # we could also move pytorch install inside docker
      python3 -m pip install "torch~=2.1.0" --index-url https://download.pytorch.org/whl/cu118
      python3 -m pip install finufft/python/cufinufft
      python3 -m pip install finufft/python/finufft

      python3 -m pip install -e .[dev]

      python3 -m pytest -k "cuda" tests/ --cov
    '''
      }
    }
  }
  post {
    failure {
      emailext subject: '$PROJECT_NAME - Build #$BUILD_NUMBER - $BUILD_STATUS',
           body: '''$PROJECT_NAME - Build #$BUILD_NUMBER - $BUILD_STATUS

Check console output at $BUILD_URL to view full results.

Building $BRANCH_NAME for $CAUSE
$JOB_DESCRIPTION

Chages:
$CHANGES

End of build log:
${BUILD_LOG,maxLines=200}
''',
           recipientProviders: [
         [$class: 'DevelopersRecipientProvider'],
           ],
           replyTo: '$DEFAULT_REPLYTO',
           to: 'bward@flatironinstitute.org'
    }
  }
}
