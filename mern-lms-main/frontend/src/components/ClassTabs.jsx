import PropTypes from "prop-types";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Box from "@mui/material/Box";
import ClassMaterial from "./ClassMaterial";
import ClassGroup from "./ClassGroup";
import ClassAssignment from "./ClassAssignment";
import ClassGrade from "./ClassGrade";
import ClassForum from "./ClassForum";
import { useDispatch, useSelector } from "react-redux";
import { setTabIndex } from "../redux/tabIndex/tabIndexSlice";
import { Stack } from "@mui/material";
// import MenuBookIcon from "@mui/icons-material/MenuBook";
// import GroupsIcon from "@mui/icons-material/Groups";
// import AssignmentIcon from "@mui/icons-material/Assignment";
// import SchoolIcon from "@mui/icons-material/School";
// import ForumIcon from "@mui/icons-material/Forum";
import {
  MenuBook as MenuBookIcon,
  Groups as GroupsIcon,
  Assignment as AssignmentIcon,
  School as SchoolIcon,
  Forum as ForumIcon,
} from '@mui/icons-material';

function CustomTabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

CustomTabPanel.propTypes = {
  children: PropTypes.node,
  index: PropTypes.number.isRequired,
  value: PropTypes.number.isRequired,
};

function a11yProps(index) {
  return {
    id: `simple-tab-${index}`,
    "aria-controls": `simple-tabpanel-${index}`,
  };
}

export default function ClassTabs() {
  const dispatch = useDispatch();
  const tabIndex = useSelector((state) => state.tabIndex.tabIndex);

  const handleChange = (event, newValue) => {
    dispatch(setTabIndex(newValue));
  };

  return (
    <Box sx={{ width: "100%" }}>
      <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Tabs
          value={tabIndex}
          onChange={handleChange}
          aria-label="basic tabs example"
          centered
          sx={{
            "& .Mui-selected": {
              color: "oklch(0.609 0.126 221.723) !important",
            },
            "& .MuiTabs-indicator": {
              backgroundColor: "oklch(0.609 0.126 221.723)",
            },
          }}
        >
          <Tab
            label={
              <Stack direction="row" spacing={1} alignItems="center">
                <MenuBookIcon />
                <span>Material</span>
              </Stack>
            }
            {...a11yProps(0)}
            sx={{ flexGrow: 1, textAlign: "center" }}
          />
          <Tab
            label={
              <Stack direction="row" spacing={1} alignItems="center">
                <GroupsIcon />
                <span>Group</span>
              </Stack>
            }
            {...a11yProps(1)}
            sx={{ flexGrow: 1, textAlign: "center" }}
          />
          <Tab
            label={
              <Stack direction="row" spacing={1} alignItems="center">
                <AssignmentIcon />
                <span>Assignment</span>
              </Stack>
            }
            {...a11yProps(2)}
            sx={{ flexGrow: 1, textAlign: "center" }}
          />
          <Tab
            label={
              <Stack direction="row" spacing={1} alignItems="center">
                <SchoolIcon />
                <span>Grade</span>
              </Stack>
            }
            {...a11yProps(3)}
            sx={{ flexGrow: 1, textAlign: "center" }}
          />
          <Tab
            label={
              <Stack direction="row" spacing={1} alignItems="center">
                <ForumIcon />
                <span>Forum</span>
              </Stack>
            }
            {...a11yProps(4)}
            sx={{ flexGrow: 1, textAlign: "center" }}
          />
        </Tabs>
      </Box>
      <CustomTabPanel value={tabIndex} index={0}>
        <ClassMaterial />
      </CustomTabPanel>
      <CustomTabPanel value={tabIndex} index={1}>
        <ClassGroup />
      </CustomTabPanel>
      <CustomTabPanel value={tabIndex} index={2}>
        <ClassAssignment />
      </CustomTabPanel>
      <CustomTabPanel value={tabIndex} index={3}>
        <ClassGrade />
      </CustomTabPanel>
      <CustomTabPanel value={tabIndex} index={4}>
        <ClassForum />
      </CustomTabPanel>
    </Box>
  );
}
