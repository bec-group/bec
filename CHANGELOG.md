# Changelog

<!--next-version-placeholder-->

## v0.34.0 (2023-09-07)

### Feature

* Added progress endpoint and message ([`ad60b78`](https://gitlab.psi.ch/bec/bec/-/commit/ad60b7821a1de36645e2c70023ea73bb7d141e39))

### Fix

* Added missing primary readings to sgalil grid ([`e52390a`](https://gitlab.psi.ch/bec/bec/-/commit/e52390a2269370f6806f93896234bc076a0731f4))

## v0.33.0 (2023-09-07)

### Feature

* Add sgalilg_grid to scan_plugins and make scantype flyscan scan possible ([`a5ba186`](https://gitlab.psi.ch/bec/bec/-/commit/a5ba186ad14283fae7c5160180a759e29f78137d))

### Fix

* File_writer and scan_ser for falcon and eiger9m and sgalil grid scan ([`cec0b34`](https://gitlab.psi.ch/bec/bec/-/commit/cec0b342f0c518bb37c4403cb55336792a192cec))
* Online fix for file writer ([`de5ba09`](https://gitlab.psi.ch/bec/bec/-/commit/de5ba09954468bf696e2aa27f00532fe7780ef27))
* Add file_writer plugin cSAXS and file_event for new file from device ([`0fdf164`](https://gitlab.psi.ch/bec/bec/-/commit/0fdf1647aaf153d480f952ff1515fda2a1a1640d))
* Add frames_per_trigger to scans and scan server ([`0c66dc3`](https://gitlab.psi.ch/bec/bec/-/commit/0c66dc33593379c7e2bee8499af8d6cecf32b761))
* Add eiger9m to cSAXS nexus file writer plugin ([`375150c`](https://gitlab.psi.ch/bec/bec/-/commit/375150ce58e00f2b6f53d713ac35cebdb087b6ad))
* Add file_writer plugin cSAXS and file_event for new file from device ([`b1f4fcc`](https://gitlab.psi.ch/bec/bec/-/commit/b1f4fccaaaec9cded2182554900ca48ceeb2fdc3))
* Add frames_per_trigger to scans and scan server ([`51c8a54`](https://gitlab.psi.ch/bec/bec/-/commit/51c8a54f01c6b5a0a09c90cb5a21e5640b3cd884))
* Add eiger9m to cSAXS nexus file writer plugin ([`8ba441f`](https://gitlab.psi.ch/bec/bec/-/commit/8ba441f55fdb9659aff12d2535799f268af1d815))

## v0.32.0 (2023-09-06)

### Feature

* Added pre_scan ([`7f23482`](https://gitlab.psi.ch/bec/bec/-/commit/7f23482b5cf273f06776e497783f44361a2cb58f))

### Fix

* Removed pre move from fly scan ([`ed095b0`](https://gitlab.psi.ch/bec/bec/-/commit/ed095b00cbebc50ebecaabc696b8aaf4a728270d))
* Removed pre move from fly scan ([`f8ad2f8`](https://gitlab.psi.ch/bec/bec/-/commit/f8ad2f8a2781fa38000c29b39772132eaa63e4ce))

### Documentation

* Added premove and enforce_sync doc ([`fd38985`](https://gitlab.psi.ch/bec/bec/-/commit/fd38985767ead15678f45ac60d0ee59bb8ee8df6))

## v0.31.0 (2023-09-05)

### Feature

* Added support for loading the service config from plugins ([`f3d3679`](https://gitlab.psi.ch/bec/bec/-/commit/f3d3679e216492d8dfaf35ff00f75520652863fc))

## v0.30.1 (2023-09-05)

### Fix

* Added sleep before polling for status ([`c8acaa4`](https://gitlab.psi.ch/bec/bec/-/commit/c8acaa4b71504a8b34c9f05f4ef6af5ab444a424))
* Removed hard-coded trigger wait; waiting for status instead ([`086c863`](https://gitlab.psi.ch/bec/bec/-/commit/086c8634e30baf4ae1b74ae61bd3f8070c69d320))

## v0.30.0 (2023-09-04)

### Feature

* Beamline check ([`cae5f61`](https://gitlab.psi.ch/bec/bec/-/commit/cae5f61924744d0358527b074958bdfe102bb2cd))
* Added preliminary version of bl_checks ([`bfa1d67`](https://gitlab.psi.ch/bec/bec/-/commit/bfa1d678735cc8dcfb303446517254290c7c7921))

## v0.29.0 (2023-09-04)

### Feature

* Added bec_plugins as source for devices ([`bbcdbc0`](https://gitlab.psi.ch/bec/bec/-/commit/bbcdbc0123566f4bea811fb9c873e059b4eb4a7c))

### Fix

* Fixed signal init ([`41282e5`](https://gitlab.psi.ch/bec/bec/-/commit/41282e57678d6a39a1f40fdf828e2fdb2ddc0193))

## v0.28.0 (2023-09-02)

### Feature

* Added progress bar based on async devices ([`11e5f96`](https://gitlab.psi.ch/bec/bec/-/commit/11e5f96b7575e0a811f45914e99ada6d2c449648))
* Added scan progress ([`9f6a044`](https://gitlab.psi.ch/bec/bec/-/commit/9f6a044fe316c804e2e4dfc34435c9eb71cd109b))
* Added xrange ([`f4f38d6`](https://gitlab.psi.ch/bec/bec/-/commit/f4f38d6deab2026177126e58cf1eac20490d9942))

### Fix

* Fixed scan_progress import ([`5eda477`](https://gitlab.psi.ch/bec/bec/-/commit/5eda477723d4dfc0387e0293713ef8e197a58f53))
* Ipython client should use default service config ([`9b89aec`](https://gitlab.psi.ch/bec/bec/-/commit/9b89aecfdc0449a9d40aae642dccf2408989c6d1))

## v0.27.0 (2023-08-31)

### Feature

* Added get_last; changed streams to stream suffix ([`e84601f`](https://gitlab.psi.ch/bec/bec/-/commit/e84601f487d4943c63a31f12b42d656dc9a4c690))

## v0.26.0 (2023-08-31)

### Feature

* Add new endpoint for async device readback ([`5535797`](https://gitlab.psi.ch/bec/bec/-/commit/5535797e1e25121d7a3997d78aa6c43eff17e086))

### Fix

* Fixed xadd for pipelines ([`d19fce7`](https://gitlab.psi.ch/bec/bec/-/commit/d19fce7d21a12eac2f8ac9b083fff464e5d0da9e))
* Bugfix ([`57c989c`](https://gitlab.psi.ch/bec/bec/-/commit/57c989cfe204a657bcefac2364a6a0ad98a77ff1))
* Adjust xadd to allow streams to expire ([`33fbded`](https://gitlab.psi.ch/bec/bec/-/commit/33fbdedd3eed52ded4eb53043bc7407997d51e4a))
* Online changes ([`9b07e0f`](https://gitlab.psi.ch/bec/bec/-/commit/9b07e0f8a2d774a9a6a07ab9faa9167585532dcd))

## v0.25.0 (2023-08-31)

### Feature

* Added support for startup scripts from plugins ([`d35caf5`](https://gitlab.psi.ch/bec/bec/-/commit/d35caf5ae40b5b46f3b2adad139cad66b3091857))

## v0.24.0 (2023-08-31)

### Feature

* Added global var service config to simplify sharing the config with other classes ([`75f1f9c`](https://gitlab.psi.ch/bec/bec/-/commit/75f1f9cd4ebc6938f2cf47103fb64eef8be57ae3))
* Added option to update the worker config directly ([`a417fd8`](https://gitlab.psi.ch/bec/bec/-/commit/a417fd8a18cadb4b480da243149c1186f3a07d88))
* Added available resource endpoint/message ([`5f5c80c`](https://gitlab.psi.ch/bec/bec/-/commit/5f5c80c2866236226dca717de0c67b32f5692ab9))

### Fix

* Fixed worker manager ([`fa62a8a`](https://gitlab.psi.ch/bec/bec/-/commit/fa62a8a9c96da44439ba71ae82d8020c8a2a0de5))

## v0.23.1 (2023-08-31)

### Fix

* Removed bec prefix from file path ([`9a3b20f`](https://gitlab.psi.ch/bec/bec/-/commit/9a3b20f085232369c9320bb8f54b93fb6b0b1686))

## v0.23.0 (2023-08-29)

### Feature

* Added device precision ([`4177fe6`](https://gitlab.psi.ch/bec/bec/-/commit/4177fe6038a10e2f285fc18c13ef6a77022b17e5))
* Added support for user scripts from plugins and home directory ([`cd59267`](https://gitlab.psi.ch/bec/bec/-/commit/cd59267e780586b002cd80c692a0f38c213f999d))

### Fix

* Fixed live table for hinted signals ([`4334567`](https://gitlab.psi.ch/bec/bec/-/commit/43345676533a402fac517fd467c98b46f35658aa))

## v0.22.0 (2023-08-24)

### Feature

* Added acquisition config and readout_time ([`f631759`](https://gitlab.psi.ch/bec/bec/-/commit/f63175941bbf7d9f5448ff58b9ea942bd2e1b9a4))

## v0.21.1 (2023-08-21)

### Fix

* Fixed bug in device config update ([`940737f`](https://gitlab.psi.ch/bec/bec/-/commit/940737fe6c8295423390a76b784a5984a93c7043))

## v0.21.0 (2023-08-20)

### Feature

* Inject device_manager based on signature ([`4eb9cf4`](https://gitlab.psi.ch/bec/bec/-/commit/4eb9cf494c805cdf751e459f0b9d0b7aa3ebee91))

## v0.20.0 (2023-08-20)

### Feature

* Added device precision to rpc base class ([`2c7b55f`](https://gitlab.psi.ch/bec/bec/-/commit/2c7b55f828f3f68ff05095a007724e499797126b))
* Added option to specify thread names ([`cae0ba2`](https://gitlab.psi.ch/bec/bec/-/commit/cae0ba2d3ea659a7de3936acdc257e1aa0991311))
* Added support for multiple queues; still WIP ([`9019cc2`](https://gitlab.psi.ch/bec/bec/-/commit/9019cc2c7443c38c47160af843eef7e3f070a25b))

### Fix

* Fixed interceptions for multiple queues ([`4e5d0da`](https://gitlab.psi.ch/bec/bec/-/commit/4e5d0da38b06f11e6abe5ce23687cdf237c9ffeb))
* Removed primary queue from init; cleanup ([`bb04271`](https://gitlab.psi.ch/bec/bec/-/commit/bb042716fecbc3035483184e494e9e4f3d2d82da))

## v0.19.0 (2023-08-20)

### Feature

* Added dap to client ([`0ea549a`](https://gitlab.psi.ch/bec/bec/-/commit/0ea549a599f4ac3dccffe7fa2f148e48a0c5d7c1))
* Add bec_worker_manager.py ([`f0ba36d`](https://gitlab.psi.ch/bec/bec/-/commit/f0ba36db869b8a0e06918ef1fd9fc44a87cbd217))
* Pluging support for data_processing ([`9e33418`](https://gitlab.psi.ch/bec/bec/-/commit/9e334185260e5f92964e1f3f5b5d6d3a86d4c1d6))

### Fix

* Remove parameters for saxs_imaging_processor ([`39c7a9c`](https://gitlab.psi.ch/bec/bec/-/commit/39c7a9c0be0a0b9861961e5443f313e11fb35748))
* Fixed dap worker for plugins ([`e2f3d8f`](https://gitlab.psi.ch/bec/bec/-/commit/e2f3d8f29ddc771798d0e2cc43f7f0d85db00fe9))

## v0.18.1 (2023-08-19)

### Fix

* Removed timeout ([`29df4ac`](https://gitlab.psi.ch/bec/bec/-/commit/29df4ac19ac189f4d7666c2c47c4539cf5e94372))
* Fixed bug in wait function for aborted move commands ([`019fcda`](https://gitlab.psi.ch/bec/bec/-/commit/019fcdaa074dcb67c84132cb038067dca8578830))

## v0.18.0 (2023-08-15)

### Feature

* Scan signature is now exported; simplified scan init ([`f35b04a`](https://gitlab.psi.ch/bec/bec/-/commit/f35b04a676a8c6aa972f031d83cb637b346d5d4f))

### Fix

* Fixed typo in round_roi_scan init; added test ([`75f2217`](https://gitlab.psi.ch/bec/bec/-/commit/75f221758f939c510a7766101cc3faa0250a0b6b))
* Fixed bug in unpack_scan_args for empty lists ([`a693f84`](https://gitlab.psi.ch/bec/bec/-/commit/a693f84816d9074a3f4664a8530d0b130702f7a2))
* Added missing file ([`f55a518`](https://gitlab.psi.ch/bec/bec/-/commit/f55a518b9103f93b54c872fb4387956cb783d5b8))

## v0.17.2 (2023-08-10)

### Fix

* Added MessageObject eq dunder ([`563c628`](https://gitlab.psi.ch/bec/bec/-/commit/563c6285092b9d8e33e8c93dea95986b87f5c67a))

## v0.17.1 (2023-08-10)

### Fix

* Fixed default config ([`8ad8d84`](https://gitlab.psi.ch/bec/bec/-/commit/8ad8d84e00a62306d43862192c8a16b09e17a17b))

## v0.17.0 (2023-08-10)

### Feature

* Added stream consumer ([`b4043e9`](https://gitlab.psi.ch/bec/bec/-/commit/b4043e970ac0d3fe2bbd6cb8d386967aefcf812d))

### Fix

* Fixed scans if redis is not available ([`b0467a8`](https://gitlab.psi.ch/bec/bec/-/commit/b0467a86aaf4741484ef0fb66e6441e742142cb5))
* Fixed scan number if redis is not available ([`8514d2d`](https://gitlab.psi.ch/bec/bec/-/commit/8514d2d6384516f53fd75d4ef671e24f32fad0f4))
* Fixed bec_service if service keys are not available ([`9b71f77`](https://gitlab.psi.ch/bec/bec/-/commit/9b71f77dacf0fe1313fe6f0c1e9de73572286b96))

## v0.16.3 (2023-08-06)

### Fix

* Catch redis connection errors ([`31efa96`](https://gitlab.psi.ch/bec/bec/-/commit/31efa96cec20540a00f0be199e8fda4fa04fdc68))
* Fixed default arg for initialize ([`b65aba8`](https://gitlab.psi.ch/bec/bec/-/commit/b65aba8a5fcdb8f2f5eeb488725144f46267f074))
* Wait for bec server should only be done for ipython, not the bec lib ([`9dfe389`](https://gitlab.psi.ch/bec/bec/-/commit/9dfe38943f2b8d6be051612de9f31ad8171f1073))
* Scanbundler sets status to running ([`d0d46ba`](https://gitlab.psi.ch/bec/bec/-/commit/d0d46ba76b1351f5431d7c93a6d6591c250563d7))

### Documentation

* Updated style; added css ([`6ec5fac`](https://gitlab.psi.ch/bec/bec/-/commit/6ec5facd0cdf0588c6545828c53ccc9e8ed29875))
* Added simple ophyd description; added file_manager description ([`48cfcb6`](https://gitlab.psi.ch/bec/bec/-/commit/48cfcb6c6242c381aea71d0e1c686d10e3fb2c1b))

## v0.16.2 (2023-08-05)

### Fix

* Fixed check_storage for already removed scan storage items ([`4a4dace`](https://gitlab.psi.ch/bec/bec/-/commit/4a4daceaf4b7c579cb4adead784f9900b675b5dc))

## v0.16.1 (2023-08-05)

### Fix

* Added thread lock to file writer ([`27e85bb`](https://gitlab.psi.ch/bec/bec/-/commit/27e85bb8b0e5afc0c70618438506727cea883253))

## v0.16.0 (2023-08-04)

### Feature

* Added support for file references and external links in the bec master file ([`9a59bdc`](https://gitlab.psi.ch/bec/bec/-/commit/9a59bdce90110fded772bf4efd84b10e019a7837))
* Added done entry to filemessage ([`2c62fd7`](https://gitlab.psi.ch/bec/bec/-/commit/2c62fd72b16cc62840daba929c1afd8dc26956d0))
* Added support for endpoints with and without suffix ([`ce0e54e`](https://gitlab.psi.ch/bec/bec/-/commit/ce0e54e561ad5ef03898e749e7333dc7535bf0d2))

### Fix

* Removed dummy link ([`de2c8ab`](https://gitlab.psi.ch/bec/bec/-/commit/de2c8ab2c51357dd23e9efbf8481fa99adb11326))
* Removed unnecessary config assignment in client ([`9360570`](https://gitlab.psi.ch/bec/bec/-/commit/93605707bd1ec1efea51407c593b25e0e5b75620))

### Documentation

* Added missing reference file ([`df19570`](https://gitlab.psi.ch/bec/bec/-/commit/df19570c9d658b35a04dbe7112c454793a8a2e54))
* Added logo ([`3c40a28`](https://gitlab.psi.ch/bec/bec/-/commit/3c40a2856c7678d14517bfcae6fe2c935756f68d))
* Fixed requirements ([`76e9342`](https://gitlab.psi.ch/bec/bec/-/commit/76e93429f6eb3851c5fabc78ff425e28b3ba2427))
* Added glossary ([`b54e56f`](https://gitlab.psi.ch/bec/bec/-/commit/b54e56fe8fd7f29b2499770c7c392cdcf7e72fe8))
* Fixed indent ([`fe07a70`](https://gitlab.psi.ch/bec/bec/-/commit/fe07a702df434714fd500fc983502e106e410bee))
* Fixed references ([`20254fb`](https://gitlab.psi.ch/bec/bec/-/commit/20254fb628206f934238f40765a3fa5d15c3274c))
* Updated developer instructions ([`823094a`](https://gitlab.psi.ch/bec/bec/-/commit/823094acb1b06074ef3180d2717986020b911b4f))
* Redesigned documentation ([`ecf3ee9`](https://gitlab.psi.ch/bec/bec/-/commit/ecf3ee93de1fd0ea0f4694150c8c07fcc21da4b5))

## v0.15.0 (2023-08-03)

### Feature

* Added option to specify config path as service config ([`1a776de`](https://gitlab.psi.ch/bec/bec/-/commit/1a776de8118de7428b0c6b4e3693eaf619651192))

### Documentation

* Updated sphinx conf file to deal with md files; added copy button ([`7f48ce6`](https://gitlab.psi.ch/bec/bec/-/commit/7f48ce6aa1f2000993a4fb31e23a3efa3c122a57))
* Minor improvements for scan_to_csv docs ([`21d371a`](https://gitlab.psi.ch/bec/bec/-/commit/21d371a80b8009e1df3c9d4148191f05a36a0abf))

## v0.14.8 (2023-07-26)

### Fix

* Adapt write_to_csv to write multiple scan_reports for context manager ([`7118863`](https://gitlab.psi.ch/bec/bec/-/commit/71188638323f27f0ae7f643a0e8b3ade12579899))

## v0.14.7 (2023-07-25)

### Fix

* Fixed build ([`4eccc99`](https://gitlab.psi.ch/bec/bec/-/commit/4eccc996694d9b260d1df40cc5b2c74ccb587dbe))

## v0.14.6 (2023-07-25)

### Fix

* Fixed bec_client install ([`bacda25`](https://gitlab.psi.ch/bec/bec/-/commit/bacda2580a47773bc4bdabc231049fb6470e7445))

## v0.14.5 (2023-07-24)

### Fix

* Fixed install ([`3f42f2f`](https://gitlab.psi.ch/bec/bec/-/commit/3f42f2f3e1d35e9d6f825a8f9865ab3dabf61be2))

## v0.14.4 (2023-07-24)

### Fix

* Added missing init files ([`1ea9764`](https://gitlab.psi.ch/bec/bec/-/commit/1ea976411d320959a7826e6f09301df90b56517a))
* Added missing init files ([`29cf132`](https://gitlab.psi.ch/bec/bec/-/commit/29cf132a06ebcec7f1e1a8f084d35da0195d4489))
* Fixed build for device_server ([`fc90bfb`](https://gitlab.psi.ch/bec/bec/-/commit/fc90bfb9aab5ef42a9c6160be71357f0df5d21bc))

## v0.14.3 (2023-07-24)

### Fix

* Fixed bec-server version ([`72fdd91`](https://gitlab.psi.ch/bec/bec/-/commit/72fdd91da495e2150463c8aa64cab1a86577289e))

## v0.14.2 (2023-07-24)

### Fix

* Fixed version update for bec-server ([`ae4673f`](https://gitlab.psi.ch/bec/bec/-/commit/ae4673fac049e7bff799efb7566ea5a8fba56c57))

## v0.14.1 (2023-07-24)

### Fix

* Update version number directly to fix pip install without -e ([`91ffa4b`](https://gitlab.psi.ch/bec/bec/-/commit/91ffa4b3c554ab4f0f038958344b81202e251433))

## v0.14.0 (2023-07-21)

### Feature

* Add new functions to save scan to dict and csv ([`effb642`](https://gitlab.psi.ch/bec/bec/-/commit/effb642a4d3a099dd05e0f3b96ac727564e01999))

### Fix

* Fix writer functions ([`fda9d07`](https://gitlab.psi.ch/bec/bec/-/commit/fda9d07e65039e833f51192d4a66a48875c3be46))
* Code update ([`86b1985`](https://gitlab.psi.ch/bec/bec/-/commit/86b198595db33e1af6b8d2a26151658118b2ebe3))

### Documentation

* Updated build dependencies ([`8dd2116`](https://gitlab.psi.ch/bec/bec/-/commit/8dd21165f2079c64ac4e738d0f84926fd60cf887))

## v0.13.3 (2023-07-21)

### Fix

* Fixed tmux launch ([`e4d7840`](https://gitlab.psi.ch/bec/bec/-/commit/e4d78402c0f0feca7d0731498b3b34701d9bc9a6))
* Fixed single env install ([`929689c`](https://gitlab.psi.ch/bec/bec/-/commit/929689cb8e7d1fccda0ab2a5a6372e2d48696193))
* Fixed bec_server install ([`2ebf580`](https://gitlab.psi.ch/bec/bec/-/commit/2ebf580ede20c594951bde73f2a570b744904509))

## v0.13.2 (2023-07-21)

### Fix

* Pip install dev environment ([`750fe66`](https://gitlab.psi.ch/bec/bec/-/commit/750fe66ed3c7c813b9ea154055f6a6f599fadc9a))

### Documentation

* Removed user api for now ([`d8fd1d0`](https://gitlab.psi.ch/bec/bec/-/commit/d8fd1d0b984f4a32d090cfbebcf9a6511f734e09))
* Fixed dependencies; added missing files ([`87e7ec2`](https://gitlab.psi.ch/bec/bec/-/commit/87e7ec2671578ffb2f5c6db1f5d98fcdebaeb61f))
* Added missing glossary file ([`2529891`](https://gitlab.psi.ch/bec/bec/-/commit/2529891a2ca39e773651c3b96d70584c55115eab))
* Improved documentation; added how tos; added glossary ([`99f0c96`](https://gitlab.psi.ch/bec/bec/-/commit/99f0c9636b36f89dc156959184cdd31d65ffee5c))

## v0.13.1 (2023-07-18)

### Fix

* Fixed bug in BECMessage str dunder ([`65e76a9`](https://gitlab.psi.ch/bec/bec/-/commit/65e76a93ceec953434e23432b9c5e912eabcb2c0))

## v0.13.0 (2023-07-14)

### Feature

* Triggering release after refactoring (file_writer_mixin) ([`e4a51b6`](https://gitlab.psi.ch/bec/bec/-/commit/e4a51b67a63bdde93c91e07e7428759c4eb44d56))

## v0.12.0 (2023-07-12)

### Feature

* Added message version 1.2 for better performance ([`f46b29a`](https://gitlab.psi.ch/bec/bec/-/commit/f46b29a2427137be86903df7da6684613698d0c7))
* Added message version 1.2 for better performance ([`fe2bd6c`](https://gitlab.psi.ch/bec/bec/-/commit/fe2bd6c935b511d26a649f89f4ba5b44ed01b7f0))

### Fix

* Fixed bundled messages for 1.2 ([`9381c7d`](https://gitlab.psi.ch/bec/bec/-/commit/9381c7d64684c332b90480aa8c7a6774baf3b5dd))
* Improvements / fixes for redis streams ([`3f09cc3`](https://gitlab.psi.ch/bec/bec/-/commit/3f09cc3cd153e629ee550072d7fc5c31100594be))
* Improvements / fixes for redis streams ([`72e4f94`](https://gitlab.psi.ch/bec/bec/-/commit/72e4f943b684e53e16ed11538d0807d012e9e357))

## v0.11.0 (2023-07-12)

### Feature

* Added redis stream methods to RedisProducer ([`e8352aa`](https://gitlab.psi.ch/bec/bec/-/commit/e8352aa606dc999f0e1bf1bd891a7852a489509d))

## v0.10.2 (2023-07-11)

### Fix

* Added missing x coords to lmfit processor ([`ddfe9df`](https://gitlab.psi.ch/bec/bec/-/commit/ddfe9df6a11f506e52f00be59f76b43c910d0504))

## v0.10.1 (2023-07-11)

### Fix

* Fixed relative path in client init; needed for pypi ([`0d9ed33`](https://gitlab.psi.ch/bec/bec/-/commit/0d9ed33a2d63e54ac12bf9cd5dcc6d4250e70bc4))

## v0.10.0 (2023-07-08)

### Feature

* Added install_bec_dev script ([`db9539a`](https://gitlab.psi.ch/bec/bec/-/commit/db9539aba203e7e299620f76dfd1f3843ebfecbd))
* Simplified bec-server interaction; removed hard-coded service config path ([`5dd1eb7`](https://gitlab.psi.ch/bec/bec/-/commit/5dd1eb7cd0ea0d401c411c9e46b8a567e58c9687))
* Added default service config ([`b1a4b4f`](https://gitlab.psi.ch/bec/bec/-/commit/b1a4b4f75cad19e849d573beb767b18c6d93a308))
* Added clis to all services; added bec_server ([`f563800`](https://gitlab.psi.ch/bec/bec/-/commit/f563800268e7047fd9baa05e48070475688b244f))

### Fix

* Added missing services to the build script ([`6d45485`](https://gitlab.psi.ch/bec/bec/-/commit/6d45485b5a83d02612595c25a3fd3ec90f0c57b6))
* Fixed bug in ipython live update ([`a6a2c28`](https://gitlab.psi.ch/bec/bec/-/commit/a6a2c28a6a111ff552277686d7455eec9cbd56d1))
* Fixed missing files ([`047082b`](https://gitlab.psi.ch/bec/bec/-/commit/047082b38b7f4145c469a76f439fcac241a92b60))
* Adjusted import routine for plugins ([`38c4c8c`](https://gitlab.psi.ch/bec/bec/-/commit/38c4c8c93e79a37314ad5579feb77455d2a5e38f))
* Fixed bug in install script ([`1a7a4d8`](https://gitlab.psi.ch/bec/bec/-/commit/1a7a4d8a745ea29af4ccdc03b6b4d608b6b18fa8))
* Fixed bug in install script ([`05bf99a`](https://gitlab.psi.ch/bec/bec/-/commit/05bf99af739b4023ad75780fe2808f71adcc508f))
* Improved tmux_launcher to handle merged and separated envs ([`088b1a4`](https://gitlab.psi.ch/bec/bec/-/commit/088b1a4a1956209c11c5a31f5c09eca8aed6b86a))

### Documentation

* Updated deployment instructions ([`390db04`](https://gitlab.psi.ch/bec/bec/-/commit/390db0442266f1d4fc36bf8beb70715ccb692eea))
* Updated documentation for new deployment ([`dfc8c92`](https://gitlab.psi.ch/bec/bec/-/commit/dfc8c9247d6b4891cdfb489be2bd3dfba5fe8f40))

## v0.9.2 (2023-07-04)

### Fix

* Added reset_device function ([`f235a17`](https://gitlab.psi.ch/bec/bec/-/commit/f235a1735f67f25eab9ae4ed746a1c101da43dc9))
* Fixed re-enabling devices ([`3f11144`](https://gitlab.psi.ch/bec/bec/-/commit/3f111442584b9abf39382620ccf137c93c89d6a8))
* Improved getattr handling for dunder methods; added comment ([`a6c49b3`](https://gitlab.psi.ch/bec/bec/-/commit/a6c49b34ad2a6960c9db57b6ab6336bb94b432d9))
* Fixed bug in client callbacks that caused rejected scans to get stuck ([`2611f5b`](https://gitlab.psi.ch/bec/bec/-/commit/2611f5b4232fed7d930b21059c2cd0e8a1098a3a))
* Fixed bug in ipython_live_updates in case of missing status messages ([`39c4323`](https://gitlab.psi.ch/bec/bec/-/commit/39c4323303287617918d7cd7101332b338026954))

## v0.9.1 (2023-07-03)

### Fix

* Fixed bug in device_manager that killed tab-completion ([`32d313a`](https://gitlab.psi.ch/bec/bec/-/commit/32d313a04feee1437b4aff547b3ba998266d78af))

## v0.9.0 (2023-07-02)

### Feature

* Add support for scan plugins set through environment vars ([`5ad0d9b`](https://gitlab.psi.ch/bec/bec/-/commit/5ad0d9bbe49c5a0aa1bed74f19caf8df553ee98e))

## v0.8.1 (2023-07-02)

### Fix

* Fixed ipython client startup script for new lib name ([`b2f5f3c`](https://gitlab.psi.ch/bec/bec/-/commit/b2f5f3c2631d749ade619fd32b3f10671f9f3f1c))

### Documentation

* Added data_processing services; changed default python version to 3.9 ([`233f682`](https://gitlab.psi.ch/bec/bec/-/commit/233f68216ff12ce223ea4024fe190e237df21afe))
* Updated doc with proper semver ([`71aa1d7`](https://gitlab.psi.ch/bec/bec/-/commit/71aa1d715a47a9b42888147611ffc8af8d46714c))

## v0.8.0 (2023-06-28)

### Feature

* Renamed primary devices to monitored devices; closes #75 ([`1370db4`](https://gitlab.psi.ch/bec/bec/-/commit/1370db4c70b08702c29e3728b8d0c3229d0188f3))

## v0.7.1 (2023-06-28)

### Fix

* Remove outdated requirements.txt files ([`f781571`](https://gitlab.psi.ch/bec/bec/-/commit/f7815714ff9c9ab6c5b697edc651c376c8052e70))
* Setup files cleanup ([`f60889a`](https://gitlab.psi.ch/bec/bec/-/commit/f60889a87e16ff767806d47bd82a988f50fb091d))

## v0.7.0 (2023-06-28)

### Feature

* Renamed bec_client_lib to bec_lib ([`a944e43`](https://gitlab.psi.ch/bec/bec/-/commit/a944e43e1a8db55959a042a8203040fa2c5484ba))

### Documentation

* Updated readme for new bec_lib ([`6e0bf12`](https://gitlab.psi.ch/bec/bec/-/commit/6e0bf12a0ae1885245961461f0bcef09ad13c2ec))

## v0.6.14 (2023-06-27)

### Fix

* Testing build ([`6849b95`](https://gitlab.psi.ch/bec/bec/-/commit/6849b9583ff0b3c5f4b49180f78b1ef612669145))

### Documentation

* Added scan server readme ([`1663087`](https://gitlab.psi.ch/bec/bec/-/commit/1663087ff8866dff31a6974474b56dd3e73ffb1d))
* Added readme for bec-client-lib ([`bd39147`](https://gitlab.psi.ch/bec/bec/-/commit/bd391470f86ece1e26b629c75341b2ee2c941da4))

## v0.6.13 (2023-06-27)

### Fix

* Added env vars ([`3d33d4b`](https://gitlab.psi.ch/bec/bec/-/commit/3d33d4bc32d2daac29cef6e71d5e0d48aba54f7e))

## v0.6.12 (2023-06-27)

### Fix

* Build test ([`899cfab`](https://gitlab.psi.ch/bec/bec/-/commit/899cfaba35fe40457635d3c8b9840da762e4b0ba))
* Build process with env var ([`3c5f351`](https://gitlab.psi.ch/bec/bec/-/commit/3c5f35166af19faa51fef75aed48a3ded0a186e4))

## v0.6.11 (2023-06-27)

### Fix

* Testing build ([`5f20c5e`](https://gitlab.psi.ch/bec/bec/-/commit/5f20c5e32d304e973bf02e496b1c0bcc6990a302))

## v0.6.10 (2023-06-27)

### Fix

* Testing build ([`d5fb551`](https://gitlab.psi.ch/bec/bec/-/commit/d5fb5511c79dfc598089d42a184fca26a35e6b3b))

## v0.6.9 (2023-06-27)

### Fix

* Fixed build script ([`5bba42a`](https://gitlab.psi.ch/bec/bec/-/commit/5bba42a898c2b8ec5735d1f059012ac60e2222a9))

## v0.6.8 (2023-06-27)

### Fix

* Testing release ([`240d402`](https://gitlab.psi.ch/bec/bec/-/commit/240d4020b80f371d3001a59fe55ac1433edb93d9))

## v0.6.7 (2023-06-27)

### Fix

* Fixed and improved setup.cfg files ([`b04a97e`](https://gitlab.psi.ch/bec/bec/-/commit/b04a97edbb4309d0364f19df528401ad29c62c9b))

### Documentation

* Improved config helper doc strings ([`08f6ff4`](https://gitlab.psi.ch/bec/bec/-/commit/08f6ff444e24395ee759f203123d9962441f59dd))

## v0.6.6 (2023-06-25)

### Fix

* Fixed file writer for empty time stamps ([`bc5fbf6`](https://gitlab.psi.ch/bec/bec/-/commit/bc5fbf651c39c562de2b2568011c47094e155017))

## v0.6.5 (2023-06-25)

### Fix

* Fixed timestamps for h5 files; closes #76 ([`36ab89e`](https://gitlab.psi.ch/bec/bec/-/commit/36ab89e51e031697f1611a1a1c5b946d3c7c1c2a))

## v0.6.4 (2023-06-23)

### Fix

* Added missing remove_device_tag function ([`a0884ce`](https://gitlab.psi.ch/bec/bec/-/commit/a0884cea22ee32026753b0cec449c7003a2b49b5))

## v0.6.3 (2023-06-23)

### Fix

* Fixed typo ([`3cc4418`](https://gitlab.psi.ch/bec/bec/-/commit/3cc44186ab8ada514c7d950bd2acbb5b03ac8e25))
* Version variable is pulled from semantic release file ([`6669bce`](https://gitlab.psi.ch/bec/bec/-/commit/6669bce3e178ca71d664adf9a7493e7ecad4589d))

## v0.6.2 (2023-06-23)

### Fix

* Fixed scan item for intermediate repr queries ([`a915a69`](https://gitlab.psi.ch/bec/bec/-/commit/a915a6906667cff85ab62e22a9bb0ec8f96a2656))
* Fixed scan item for intermediate repr queries ([`9decff2`](https://gitlab.psi.ch/bec/bec/-/commit/9decff27a74af7d84f41ddd8f9b3585e1d353a88))

## v0.6.1 (2023-06-23)

### Fix

* Fixed monitor scan for numpy v1.25 ([`870c033`](https://gitlab.psi.ch/bec/bec/-/commit/870c03344cd55d22a89d236d88ec60e7677ed20e))

### Documentation

* Improved doc strings for scans ([`25fe364`](https://gitlab.psi.ch/bec/bec/-/commit/25fe3641442f1fe31000685664881aaa01c9cfb3))

## v0.6.0 (2023-06-22)

### Feature

* Add to_pandas method to scan items ([`858bb78`](https://gitlab.psi.ch/bec/bec/-/commit/858bb7816d02e0326492cc6d53a18d3b4fa646e9))

## v0.5.0 (2023-06-20)

### Feature

* Added bec data processing service ([`17213da`](https://gitlab.psi.ch/bec/bec/-/commit/17213da46b236cb5ff7155890e4319308350ba4c))
* Added dap message and endpoint ([`e1aa5e1`](https://gitlab.psi.ch/bec/bec/-/commit/e1aa5e199b10cf9c7570967c01a5f3b48bfe1fc6))

### Documentation

* Add commit message info to readme ([`2d8038b`](https://gitlab.psi.ch/bec/bec/-/commit/2d8038bac7ecb3563025ceddfe08b177f94bdf6c))

## v0.4.9 (2023-06-19)

### Fix

* Raise when device does not exist; added str dunder for devices ([`12e2d29`](https://gitlab.psi.ch/bec/bec/-/commit/12e2d29dad71c11586cee06cb0688557c3cb4bb2))

### Documentation

* Added more doc strings ([`c8cc156`](https://gitlab.psi.ch/bec/bec/-/commit/c8cc15632d4221877a19296bb7d8b7742c1e4ccd))

## v0.4.8 (2023-06-19)

### Fix

* Removed changelog dependency ([`2be1c67`](https://gitlab.psi.ch/bec/bec/-/commit/2be1c67cbc7a025314a665c6d272ac2874e02fee))

## v0.4.7 (2023-06-19)

### Fix

* Fixed typo ([`f59e73c`](https://gitlab.psi.ch/bec/bec/-/commit/f59e73cbb11c1242115b1b42b97cbeb0f0f6252b))
* Fixed weird semantic-release syntax ([`eabb210`](https://gitlab.psi.ch/bec/bec/-/commit/eabb210b6e0e68269854026a8a71c07cd9274c04))

## v0.4.6 (2023-06-19)

### Fix

* Removed pypi upload ([`0b28025`](https://gitlab.psi.ch/bec/bec/-/commit/0b280253701b5e49ea37512cda6bad888e4b8149))

## v0.4.5 (2023-06-19)

### Fix

* Removed build ([`1171e65`](https://gitlab.psi.ch/bec/bec/-/commit/1171e651959df0b07f4e9ce096a8b6e4e77b132b))

## v0.4.4 (2023-06-19)

### Fix

* Disabled upload to repository ([`2e56468`](https://gitlab.psi.ch/bec/bec/-/commit/2e564681016b1b369c65087ab447444eca8a2c9a))

## v0.4.3 (2023-06-19)

### Fix

* Pull from origin ([`6c659a9`](https://gitlab.psi.ch/bec/bec/-/commit/6c659a94c4dbd5b7a4a3718c08b6fd1b117c3602))
* Checkout master instead of commit ([`33e0669`](https://gitlab.psi.ch/bec/bec/-/commit/33e0669323e3fb01d079fb56018349d190537101))
* Added git pull ([`7e77444`](https://gitlab.psi.ch/bec/bec/-/commit/7e77444a70647706ae448186fb44c64a3622880c))
* Fixed domain name ([`a3c2e5f`](https://gitlab.psi.ch/bec/bec/-/commit/a3c2e5ff85dbdd6badc182828ee85b6e01dc6377))
* Added hvcs domain ([`32856c5`](https://gitlab.psi.ch/bec/bec/-/commit/32856c50047c6a91a10f2a3666738dc6b7f16737))

## v0.4.2 (2023-06-19)

### Fix

* Changed semantic-release version to publish ([`5e12ef4`](https://gitlab.com/bec/bec/-/commit/5e12ef43171b6b75abd666aabd9060f132e53fce))
* Delete all local tags before adding new ones ([`b8d71f5`](https://gitlab.com/bec/bec/-/commit/b8d71f5cabf80d099bf76687758f398bff9214e1))
* Only run semver on master; added git tag log ([`b63d128`](https://gitlab.com/bec/bec/-/commit/b63d128cefb50dfdb52328bae7032a22cd9d5934))
* Np.vstack must receive tuple ([`3286d46`](https://gitlab.com/bec/bec/-/commit/3286d46163e4ce7d262c170a0d04a59f287b40c1))

## v0.4.1 (2023-06-19)


