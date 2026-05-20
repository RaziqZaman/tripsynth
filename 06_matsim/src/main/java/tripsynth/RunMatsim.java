package tripsynth;

import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.Controler;
import org.matsim.core.scenario.ScenarioUtils;

public final class RunMatsim {
    private RunMatsim() {
    }

    public static void main(String[] args) {
        if (args.length != 1) {
            throw new IllegalArgumentException("usage: RunMatsim <config.xml>");
        }
        Config config = ConfigUtils.loadConfig(args[0]);
        Controler controler = new Controler(ScenarioUtils.loadScenario(config));
        controler.run();
    }
}
