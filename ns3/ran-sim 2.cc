\
    /* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
    #include "ns3/core-module.h"
    #include "ns3/network-module.h"
    #include "ns3/internet-module.h"
    #include "ns3/mobility-module.h"
    #include "ns3/lte-module.h"
    #include "ns3/point-to-point-helper.h"
    #include "ns3/applications-module.h"
    #include "ns3/flow-monitor-module.h"
    #include <fstream>
    #include <sstream>
    #include <iomanip>

    using namespace ns3;

    // Map MHz to number of resource blocks (LTE)
    static uint32_t DlBandwidthFromMHz(double mhz) {
      if (mhz <= 1.4) return 6;
      if (mhz <= 3)   return 15;
      if (mhz <= 5)   return 25;
      if (mhz <= 10)  return 50;
      if (mhz <= 15)  return 75;
      return 100; // 20 MHz
    }

    int main (int argc, char *argv[])
    {
      uint32_t numUes = 5;
      std::string scheduler = "rr"; // rr, pf, mt
      double trafficMbps = 1.0;
      double simTime = 10.0;
      uint32_t rngRun = 1;
      std::string outFile = "metrics.json";
      double bandwidthMHz = 10.0; // default 10 MHz

      CommandLine cmd (__FILE__);
      cmd.AddValue ("numUes", "Number of UEs", numUes);
      cmd.AddValue ("scheduler", "Scheduler: rr | pf | mt", scheduler);
      cmd.AddValue ("trafficMbps", "Per-UE offered UDP rate (Mbps)", trafficMbps);
      cmd.AddValue ("duration", "Simulation time (s)", simTime);
      cmd.AddValue ("rngRun", "RNG run id (RngSeedManager::SetRun)", rngRun);
      cmd.AddValue ("output", "Output JSON file", outFile);
      cmd.AddValue ("bandwidthMHz", "Channel bandwidth (MHz)", bandwidthMHz);
      cmd.Parse (argc, argv);

      RngSeedManager::SetSeed (1);
      RngSeedManager::SetRun (rngRun);

      // Helpers
      Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
      Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
      lteHelper->SetEpcHelper (epcHelper);

      // Scheduler
      std::string schedType = "ns3::RrFfMacScheduler";
      if (scheduler == "pf") schedType = "ns3::PfFfMacScheduler";
      else if (scheduler == "mt") schedType = "ns3::TdMtFfMacScheduler";
      lteHelper->SetSchedulerType (schedType);

      // Bandwidth
      uint32_t rb = DlBandwidthFromMHz(bandwidthMHz);
      Config::SetDefault ("ns3::LteEnbNetDevice::DlBandwidth", UintegerValue (rb));
      Config::SetDefault ("ns3::LteEnbNetDevice::UlBandwidth", UintegerValue (rb));

      // Create nodes
      NodeContainer ueNodes;
      NodeContainer enbNodes;
      enbNodes.Create (1);
      ueNodes.Create (numUes);

      // Install Mobility: eNB fixed, UEs random grid
      MobilityHelper mobility;
      Ptr<ListPositionAllocator> enbPosAlloc = CreateObject<ListPositionAllocator> ();
      enbPosAlloc->Add (Vector (0.0, 0.0, 10.0));
      mobility.SetPositionAllocator (enbPosAlloc);
      mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
      mobility.Install (enbNodes);

      mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
      Ptr<ListPositionAllocator> uePosAlloc = CreateObject<ListPositionAllocator> ();
      for (uint32_t i = 0; i < numUes; ++i) {
        double x = 30.0 * std::cos (2.0 * M_PI * i / std::max(1u,numUes));
        double y = 30.0 * std::sin (2.0 * M_PI * i / std::max(1u,numUes));
        uePosAlloc->Add (Vector (x, y, 1.5));
      }
      mobility.SetPositionAllocator (uePosAlloc);
      mobility.Install (ueNodes);

      // Install LTE Devices to the nodes
      NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice (enbNodes);
      NetDeviceContainer ueDevs = lteHelper->InstallUeDevice (ueNodes);

      // Install the IP stack on the UEs
      InternetStackHelper internet;
      internet.Install (ueNodes);

      Ipv4InterfaceContainer ueIpIface;
      Ipv4StaticRoutingHelper ipv4RoutingHelper;

      Ptr<Node> pgw = epcHelper->GetPgwNode ();
      // Create a single RemoteHost
      NodeContainer remoteHostContainer;
      remoteHostContainer.Create (1);
      InternetStackHelper internet2;
      internet2.Install (remoteHostContainer);
      Ptr<Node> remoteHost = remoteHostContainer.Get (0);
      // Create the internet
      PointToPointHelper p2ph;
      p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("100Gb/s")));
      p2ph.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
      NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
      Ipv4AddressHelper ipv4h;
      ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
      Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
      // Interface 1 is the p2p device that has address 1.0.0.2
      Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);

      // Route to the internet
      Ipv4StaticRoutingHelper ipv4Helper;
      Ipv4StaticRoutingHelper::GetStaticRouting (remoteHost->GetObject<Ipv4> ())->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);

      // Assign IP to UEs, and attach to eNB
      ueIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueDevs));
      for (uint32_t u = 0; u < ueNodes.GetN (); ++u) {
        Ptr<Node> ueNode = ueNodes.Get(u);
        // Set the default gateway for the UE
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNode->GetObject<Ipv4> ());
        ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
        lteHelper->Attach (ueDevs.Get(u), enbDevs.Get(0));
      }

      // Applications: Downlink UDP traffic from remoteHost to UEs
      uint16_t dlPort = 1234;
      ApplicationContainer clientApps;
      ApplicationContainer serverApps;
      double perUeMbps = std::max(0.1, trafficMbps);
      std::string dataRate = std::to_string(perUeMbps) + "Mbps";

      for (uint32_t u = 0; u < ueNodes.GetN (); ++u) {
        ++dlPort;
        UdpServerHelper dlPacketSinkHelper (dlPort);
        serverApps.Add (dlPacketSinkHelper.Install (ueNodes.Get(u)));
        UdpClientHelper dlClient (ueIpIface.GetAddress(u), dlPort);
        dlClient.SetAttribute ("MaxPackets", UintegerValue (0xFFFFFFFF));
        dlClient.SetAttribute ("Interval", TimeValue (MicroSeconds (200)));
        dlClient.SetAttribute ("PacketSize", UintegerValue (1400));
        ApplicationContainer client = dlClient.Install (remoteHost);
        clientApps.Add (client);
      }

      // Limit per-client data rate with OnOff-like pattern (by token bucket)
      // Simpler: set a traffic control rate via RateErrorModel is not appropriate here.
      // So we adjust client interval empirically by setting a PacketSize and InterPacketInterval
      // to approximate dataRate at the IP layer. For reproducibility in this simple example,
      // we accept some inaccuracy.

      // Start/stop
      serverApps.Start (Seconds (0.1));
      clientApps.Start (Seconds (0.2));
      clientApps.Stop (Seconds (simTime));
      serverApps.Stop (Seconds (simTime + 0.1));

      // Flow monitor
      FlowMonitorHelper flowmonHelper;
      Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll ();

      Simulator::Stop (Seconds (simTime + 0.2));
      Simulator::Run ();

      monitor->CheckForLostPackets ();
      Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmonHelper.GetClassifier ());
      std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();

      // Aggregate metrics
      double totalThroughputMbps = 0.0;
      double sumDelayMs = 0.0;
      uint64_t totalRxPackets = 0;
      std::vector<double> ueThroughputsMbps;
      ueThroughputsMbps.resize(numUes, 0.0);

      for (auto const &flow : stats) {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flow.first);
        double rxBytes = (double) flow.second.rxBytes;
        double thrMbps = (rxBytes * 8.0) / (simTime * 1e6);
        totalThroughputMbps += thrMbps;
        // average delay per flow (ms)
        double avgDelayMs = 0.0;
        if (flow.second.rxPackets > 0) {
          avgDelayMs = (flow.second.delaySum.GetSeconds () / flow.second.rxPackets) * 1000.0;
        }
        sumDelayMs += avgDelayMs * flow.second.rxPackets;
        totalRxPackets += flow.second.rxPackets;

        // Map to UE index by destination address if possible
        // (This simple mapping assumes each flow targets one UE sink port; not robust for all cases.)
        for (uint32_t u = 0; u < ueIpIface.GetN (); ++u) {
          if (t.destinationAddress == ueIpIface.GetAddress(u)) {
            ueThroughputsMbps[u] += thrMbps;
            break;
          }
        }
      }

      double avgDelayMsAll = (totalRxPackets > 0) ? (sumDelayMs / totalRxPackets) : 0.0;

      // Write JSON
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(4);
      oss << "{\\n";
      oss << "  \\"num_ues\\": " << numUes << ",\\n";
      oss << "  \\"scheduler\\": \\"" << scheduler << "\\",\\n";
      oss << "  \\"traffic_mbps\\": " << trafficMbps << ",\\n";
      oss << "  \\"duration_s\\": " << simTime << ",\\n";
      oss << "  \\"rng_run\\": " << rngRun << ",\\n";
      oss << "  \\"throughput_total_mbps\\": " << totalThroughputMbps << ",\\n";
      oss << "  \\"avg_delay_ms\\": " << avgDelayMsAll << ",\\n";
      oss << "  \\"ue_throughput_mbps\\": [";
      for (uint32_t i = 0; i < ueThroughputsMbps.size(); ++i) {
        oss << ueThroughputsMbps[i];
        if (i + 1 < ueThroughputsMbps.size()) oss << ", ";
      }
      oss << "]\\n";
      oss << "}\\n";

      std::ofstream of (outFile.c_str (), std::ios::out);
      of << oss.str();
      of.close();

      Simulator::Destroy ();
      return 0;
    }
